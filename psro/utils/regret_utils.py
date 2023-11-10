"""Utilities for computing empirical regret."""
from typing import Mapping

import numpy as np
from marl import types

from psro import strategy


def regret(game_matrix: np.ndarray, solution: strategy.JointStrategy) -> Mapping[types.PlayerID, float]:
  """Compute empirical regret.

  TODO: This only works for square matrices rn.

  Args:
    game_matrix:
      (P1, P2, ... PN, NP), where Pi is player i's strategy set and NP is number of players.
    solution:
  """
  regrets = {}
  soln_payoffs = solution_payoffs(game_matrix, solution)

  for player_id, soln_payoff in soln_payoffs.items():
    all_but_current_player = dict(solution)
    del all_but_current_player[player_id]
    deviation_payoff = np.max(reduce_payoffs(game_matrix, all_but_current_player)[..., player_id])
    regrets[player_id] = deviation_payoff - soln_payoff
  return regrets


def solution_payoffs(game_matrix: np.ndarray, solution: strategy.JointStrategy) -> Mapping[types.PlayerID, float]:
  """Compute the payoffs for a solution."""
  num_players = game_matrix.shape[-1]

  # Swap the utilities axis (NP) to be the leading axis.
  new_shape = list(range(game_matrix.ndim))
  new_shape = [game_matrix.ndim - 1] + new_shape[:-1]
  game_matrix = np.transpose(game_matrix, new_shape)  # (NP, P1, P2, ... PN).

  # Compute the solution payoffs.
  joint_soln = [solution[i] for i in range(num_players)]  # (NP,).
  soln_payoffs = game_matrix
  for soln in reversed(joint_soln):
    soln_payoffs = soln_payoffs @ soln
  return dict(enumerate(soln_payoffs))


def reduce_payoffs(game_matrix: np.ndarray, solution: strategy.JointStrategy) -> Mapping[types.PlayerID, float]:
  """Collapse the payoffs in a game matrix with a partial solution."""
  # Swap the utilities axis (NP) to be the leading axis.
  new_shape = list(range(game_matrix.ndim))
  new_shape = [game_matrix.ndim - 1] + new_shape[:-1]

  # Move missing axes to the leading axes, safe NP.
  included = [pid for pid in range(len(game_matrix.shape[:-1])) if pid in solution]
  excluded = [pid for pid in range(len(game_matrix.shape[:-1])) if pid not in solution]
  new_shape = [game_matrix.ndim - 1] + excluded + included
  game_matrix = np.transpose(game_matrix, new_shape)  # (NP, P1, P2, ... PN).

  # Collapse the payoff matrix on included player strategies.
  included_players = reversed(sorted(solution.keys()))
  for player in included_players:
    game_matrix = game_matrix @ solution[player]

  # Place the utilities axis (NP) as the trailing axis.
  new_shape = list(range(game_matrix.ndim))
  new_shape = new_shape[1:] + [0]
  game_matrix = np.transpose(game_matrix, new_shape)  # (NP, P1, P2, ... PN).

  return game_matrix
