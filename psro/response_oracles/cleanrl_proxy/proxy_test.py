"""Test for `TODO`."""
import functools
import os
import pathlib
import tempfile
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pyspiel
from absl.testing import absltest, parameterized
from marl import bots, individuals
from marl.games import openspiel_proxy

from psro import core, strategy
from psro.response_oracles.cleanrl_proxy import proxy
from psro.response_oracles.cleanrl_proxy.handlers import base_handler, dqn_jax
from psro.utils.wrappers import game_to_env, gym_wrapper


class _StubHandler(base_handler.Handler):
  """Handles building patches and the resultant bot for specific algorithms."""

  def __init__(self, job: core.ResponseOracleJob) -> None:
    """Initializer."""
    super().__init__()
    self.name = "dummy"
    self._called_patches = False
    self._called_build = False
    self._expected_job = job

  def build_patches(self, job: core.ResponseOracleJob):
    """Build patches that enable the main program to be run as a subroutine."""
    assert job.learner_id == self._expected_job.learner_id
    assert job.game_ctor == self._expected_job.game_ctor
    assert job.solution == self._expected_job.solution
    assert job.epoch_dir == self._expected_job.epoch_dir
    self._called_patches = True
    return []

  def build_bot(self, job: core.ResponseOracleJob, run_dir: pathlib.Path) -> individuals.Bot:
    """Build a bot from a completed job."""
    assert run_dir.parts[-1] == "run_dir"
    assert job.learner_id == self._expected_job.learner_id
    assert job.game_ctor == self._expected_job.game_ctor
    assert job.solution == self._expected_job.solution
    assert job.epoch_dir == self._expected_job.epoch_dir
    self._called_build = True
    return "BOT"


def _stub_importlib(path: str):
  """Stub to patch in for `importlib` to prevent actually calling CleanRL."""
  del path

  class _Module:
    """Stub CleanRL module."""

    def main(self):
      """Pretends to be `__main__`."""
      pass

  return _Module


class CleanRLProxyTest(parameterized.TestCase):
  """Test suite for `CleanRLProxy`."""

  def setUp(self):
    """Set-up before a test."""
    self._temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self._epoch_dir = pathlib.Path(self._temp_dir.name)

  def tearDown(self):
    """Tear-down after a test."""
    self._temp_dir.cleanup()

  @patch("importlib.import_module", _stub_importlib)
  def test_proxy(self):
    """Basic regression test of CleanRLProxy."""
    os.makedirs(str(self._epoch_dir / "player_0" / "runs" / "run_dir"))
    job = core.ResponseOracleJob(
        learner_id=0,
        players={0: None},
        game_ctor=lambda: None,
        solution=None,
        epoch_dir=self._epoch_dir,
    )
    oracle = proxy.CleanRLProxy(_StubHandler(job))
    self.assertEqual("BOT", oracle(job)[1])
    # pylint: disable=protected-access
    self.assertTrue(oracle._handler._called_patches)
    self.assertTrue(oracle._handler._called_build)
    # pylint: enable=protected-access

  def test_dqn_cartpole(self):
    """Regression test for DQN on Gym's CartPole-v1."""
    job = core.ResponseOracleJob(
        learner_id=0,
        players={0: None},
        game_ctor=functools.partial(
            game_to_env.EnvToGame,
            env=gym_wrapper.FromGymnasium(env=gym.make(id="CartPole-v1", max_episode_steps=100)),
        ),
        solution=None,
        epoch_dir=self._epoch_dir,
    )
    oracle = proxy.CleanRLProxy(
        dqn_jax.DQNJAXHandler(functools.partial(dqn_jax.QNetwork, action_dim=2), total_timesteps=1_000)
    )
    bot = oracle(job)[1]

    game = game_to_env.EnvToGame(gym_wrapper.FromGymnasium(env=gym.make(id="CartPole-v1", max_episode_steps=100)))
    timesteps = game.reset()
    agent_state = bot.episode_reset(timesteps[0])
    agent_state, action = bot.step(agent_state, timesteps[0])
    timesteps = game.step({0: action})
    _ = bot.step(agent_state, timesteps[0])

  def test_dqn_kuhn_poker(self):
    """Regression test for DQN on OpenSpiel's Kuhn Poker."""
    job = core.ResponseOracleJob(
        learner_id=0,
        players={
            0: strategy.Strategy([bots.RandomActionBot(2)], mixture=[1.0]),
            1: strategy.Strategy([bots.RandomActionBot(2)], mixture=[1.0]),
        },
        game_ctor=functools.partial(
            openspiel_proxy.OpenSpielProxy,
            game=pyspiel.load_game("kuhn_poker"),
            include_full_state=True,
        ),
        solution={0: np.asarray([1.0]), 1: np.asarray([1.0])},
        epoch_dir=self._epoch_dir,
    )
    oracle = proxy.CleanRLProxy(
        dqn_jax.DQNJAXHandler(
            functools.partial(dqn_jax.QNetwork, action_dim=2),
            pyspiel_game=True,
            total_timesteps=1_000,
        )
    )
    bot = oracle(job)[1]

    game = openspiel_proxy.OpenSpielProxy(game=pyspiel.load_game("kuhn_poker"), include_full_state=True)
    timesteps = game.reset()
    agent_state = bot.episode_reset(timesteps[0])
    agent_state, action = bot.step(agent_state, timesteps[0])
    timesteps = game.step({0: action})


if __name__ == "__main__":
  absltest.main()
