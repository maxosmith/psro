"""Patches to inject into DQN JAX."""
import dataclasses
import functools
import pathlib
from typing import Callable

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
from marl import individuals
from marl.services import snapshotter

from psro import core
from psro.response_oracles.cleanrl_proxy import bots as cleanrl_bots
from psro.response_oracles.cleanrl_proxy.handlers import base_handler
from psro.utils.wrappers import game_to_env, gym_wrapper

ALG_NAME = "dqn_jax"


@dataclasses.dataclass
class Args:
  """See cleanrl.dqn_jax for details."""

  exp_name: str = "cleanRL_dqn"
  seed: int = 1

  # Evaluation.
  save_model: bool = False
  capture_video: bool = False

  # Weights and Biases.
  track: bool = False
  wandb_project_name: str = "cleanRL"
  wandb_entity: str | None = None

  # Hugging Face.
  upload_model: bool = False
  hf_entity: str = ""

  # Unused.
  env_id: str = "CartPole-v1"

  # Algorithm specific arguments.
  total_timesteps: int = 500_000
  learning_rate: float = 2.5e-4
  num_envs: int = 1
  buffer_size: int = 10_000
  gamma: float = 0.99
  tau: float = 1.0
  target_network_frequency: int = 500
  batch_size: int = 128
  start_e: float = 1
  end_e: float = 0.05
  exploration_fraction: float = 0.5
  learning_starts: int = 10_000
  train_frequency: int = 10


class QNetwork(nn.Module):
  """Example QNetwork from CleanRL."""

  layer_sizes: list[int]
  action_dim: int

  @nn.compact
  def __call__(self, observation: jnp.ndarray):
    """Forward pass."""
    x = observation
    for size in self.layer_sizes:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    x = nn.Dense(self.action_dim)(x)
    return x


class DQNJAXHandler(base_handler.Handler):
  """."""

  def __init__(
      self,
      q_network_ctor: Callable[[], nn.Module],
      seed: int = 42,
      pyspiel_game: bool = False,
      total_timesteps: int = 500_000,
      learning_rate: float = 2.5e-4,
      num_envs: int = 1,
      buffer_size: int = 10_000,
      gamma: float = 0.99,
      tau: float = 1.0,
      target_network_frequency: int = 500,
      batch_size: int = 128,
      start_e: float = 1,
      end_e: float = 0.05,
      exploration_fraction: float = 0.5,
      learning_starts: int = 10_000,
      train_frequency: int = 10,
  ) -> None:
    """Initializer."""
    super().__init__()
    self.name = "dqn_jax"
    self.q_network_ctor = q_network_ctor
    self.pyspiel_game = pyspiel_game
    self.args = Args(
        seed=seed,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        num_envs=num_envs,
        buffer_size=buffer_size,
        gamma=gamma,
        tau=tau,
        target_network_frequency=target_network_frequency,
        batch_size=batch_size,
        start_e=start_e,
        end_e=end_e,
        exploration_fraction=exploration_fraction,
        learning_starts=learning_starts,
        train_frequency=train_frequency,
    )

  def build_patches(self, job: core.ResponseOracleJob):
    """Build patches that enable the main program to be run as a subroutine."""

    def _make_env(*args, **kwargs):
      """Overwrite making the environment."""
      del args, kwargs

      other_players = {}
      for player_id, player_strategy in job.players.items():
        if player_id == job.learner_id:
          continue
        player_strategy.mixture = job.solution[player_id]
        other_players[player_id] = player_strategy

      env = game_to_env.GameToEnv(
          game=job.game_ctor(),
          player_id=job.learner_id,
          other_players=other_players,
      )
      env = gym_wrapper.ToGymnasium(env)
      return functools.partial(gym.wrappers.RecordEpisodeStatistics, env=env)

    def _parse_args():
      """Overwrite parsing command-line arguments."""
      args = self.args
      args.save_model = True
      # Env ID is only used to set the run name and in calls to `make_env`.
      # Modified here so that the run names are descriptive.
      args.env_id = args.exp_name
      return args

    def _build_q_network(*args, **kwargs):
      """Overwrite building the QNetwork."""
      del args, kwargs  # Keep API compatible, handle construction here.
      return self.q_network_ctor()

    return [
        (f"cleanrl.{ALG_NAME}.make_env", {"new": _make_env}),
        (f"cleanrl.{ALG_NAME}.parse_args", {"new": _parse_args}),
        (f"cleanrl.{ALG_NAME}.QNetwork", {"new": _build_q_network}),
    ]

  def build_bot(
      self,
      job: core.ResponseOracleJob,
      run_dir: pathlib.Path,
  ) -> individuals.Bot:
    """Build a bot from a completed job."""
    # Generate the shape of the network's parameters so that we can deserialize the saved params.
    observation = job.game_ctor().observation_specs()[job.learner_id]
    if self.pyspiel_game:
      observation = observation["info_state"].generate_value()
    else:
      observation = observation.generate_value()
    q_network = self.q_network_ctor()
    q_key = jax.random.PRNGKey(42)  # This is only used to load params.
    params = q_network.init(q_key, observation)

    with open(run_dir / f"{self.args.exp_name}.cleanrl_model", "rb") as f:
      params = flax.serialization.from_bytes(params, f.read())

    snapshot = snapshotter.Snapshot(
        ctor=self.q_network_ctor,
        ctor_kwargs={},
        trace_kwargs={"observation": observation},
        params=params,
    )

    if self.pyspiel_game:
      return cleanrl_bots.OpenSpielValueProxyBot(q_network_snapshot=snapshot)
    else:
      return cleanrl_bots.ValueProxyBot(q_network_snapshot=snapshot)
