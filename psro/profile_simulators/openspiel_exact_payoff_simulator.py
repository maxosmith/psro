"""Compute player payoffs using OpenSpiel."""
import dataclasses

import numpy as np
import pyspiel
from marl import types
from open_spiel.python.algorithms import exploitability as openspiel_exploitability

from psro import core, strategy
from psro.response_oracles.openspiel_br import utils


def payoffs(game: pyspiel.Game, players: strategy.JointStrategy):
  """Computes the per-player payoffs."""
  aggr_policy = utils.aggregate_joint_strategy(game, players)
  root_state = game.new_initial_state()
  # pylint: disable=protected-access
  per_player_payoffs = openspiel_exploitability._state_values(root_state, game.num_players(), aggr_policy)
  # pylint: enable=protected-access
  return dict(enumerate(per_player_payoffs))


@dataclasses.dataclass
class ProfileSimulator:
  """."""

  game: pyspiel.Game | str

  def __post_init__(self):
    """Post initializer."""
    if isinstance(self.game, str):
      self.game = pyspiel.load_game(self.game)

  def __call__(self, job: core.SimulationJob) -> tuple[strategy.Profile, types.Tree]:
    """."""
    if job.num_episodes != 1:
      raise ValueError("OpenSpiel payoff simulator is exact. Only one estimation is needed per profile.")

    players = {}
    for player_id, player_strategy in job.players.items():
      mixture = np.zeros_like(player_strategy.mixture)
      mixture[job.profile[player_id]] = 1.0
      player_strategy.mixture = mixture
      players[player_id] = player_strategy

    results = payoffs(self.game, players)
    results = {k: np.array([v]) for k, v in results.items()}
    return (job.profile, results)
