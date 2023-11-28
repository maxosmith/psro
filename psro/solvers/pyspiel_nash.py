"""Proxy for OpenSpiels Nash solver."""
from typing import Sequence

import numpy as np
import pyspiel
import sox
from open_spiel.python.algorithms import lp_solver

from psro import strategy


class Nash:
  """Computes a uniform distribution over policies."""

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Compute uniform for the given payoff matrix.

    Args:
      payoffs: Payoff matrix [P1, P2, ...., PN, NumPlayers].

    Returns:
      Computed solution.
    """
    payoff_tensors = sox.array_utils.unstack(payoffs, axis=-1)
    payoff_tensors = [x.tolist() for x in payoff_tensors]
    if len(payoff_tensors) != 2:
      raise NotImplementedError(
          "nash_strategy solver works only for 2p zero-sum"
          f"games, but was invoked for a {len(payoff_tensors)} player game."
      )
    nash_prob_1, nash_prob_2, _, _ = lp_solver.solve_zero_sum_matrix_game(pyspiel.create_matrix_game(*payoff_tensors))
    result = [renormalize(np.array(nash_prob_1).reshape(-1)), renormalize(np.array(nash_prob_2).reshape(-1))]
    return [dict(enumerate(result))]


def renormalize(probabilities):
  """Replaces all negative entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities
