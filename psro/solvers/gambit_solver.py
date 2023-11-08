"""Gambit interface for Nash solving.

This is meant to be a thin wrapper around `pygambit`, abiding by a
common interface for empirical-game solvers, and removing the need
to learn the details of Gambit for practioners.
"""
from enum import Enum
from typing import Sequence

import numpy as np
import pygambit
from pygambit import nash

from psro import strategy


class NashSolver(Enum):
  """Gambit Nash equilibrium solving algorithms.

  References:
    - https://gambitproject.readthedocs.io/en/latest/pyapi.html#calling-command-line-tools
    - https://gambitproject.readthedocs.io/en/latest/pyapi.html#calling-internally-linked-libraries
  """

  LCP = nash.lcp_solve
  LP = nash.lp_solve
  ENUM_PURE = nash.enumpure_solve
  ENUM_MIXED = nash.enummixed_solve
  SIMP_DIV = nash.simpdiv_solve
  GNM = nash.gnm_solve
  IPA = nash.ipa_solve


def _to_decimal(x) -> pygambit.Decimal:
  """Convert a number into pygambit numerics."""
  return pygambit.Decimal(float(x))


_array_to_decimal = np.vectorize(_to_decimal)


def payoff_matrix_to_gambit_game(payoffs: np.ndarray) -> pygambit.Game:
  """Convert a payoff matrix into a pygambit Game instance.

  Args:
    payoffs: A numpy array where the last dimension indicates the
      payoffs for each player. For example, for a 2x2 game for two players,
      the shape would be (2, 2, 2).

  Returns:
    pygambit.Game: A game instance with the given payoffs.

  Raises:
    ValueError: If the payoff matrix has an unexpected shape.
  """
  payoffs = _array_to_decimal(payoffs)
  game = pygambit.Game.from_arrays(*(payoffs[..., i] for i in range(payoffs.shape[-1])))
  return game


def assert_valid_payoff_matrix(payoffs: np.ndarray):
  """Assert that a payoff matrix is a valid shape.

  Args:
    payoffs: A numpy array where the last dimension indicates the
      payoffs for each player. For example, for a 2x2 game for two players,
      the shape would be (2, 2, 2).

  Raises:
    ValueError: If the payoff matrix has an unexpected shape.
  """
  if len(payoffs.shape) < 2:
    raise ValueError("Payoff matrices need at least 2 dimensions.")
  num_players = payoffs.shape[-1]
  if len(payoffs.shape[:-1]) != (num_players):
    raise ValueError(f"Invalid payoffs matrix shape: {payoffs.shape}")


class GambitSolver:
  """Game solving through the `gambit` library.

  Args:
    algorithm:
  """

  def __init__(self, algorithm: NashSolver = NashSolver.LCP):
    """Initializer."""
    self._algorithm = algorithm

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Calculate an equilibrium from a game matrix.

    Args:
        payoffs: Game matrix of shape [|Pi_0|, |Pi_1|, ..., |Pi_{n-1}|, n].

    Returns:
        Mixed-strategy for each agent.
    """
    del kwargs
    assert_valid_payoff_matrix(payoffs)
    game = payoff_matrix_to_gambit_game(payoffs)
    solutions = self._algorithm(game, rational=False)

    # Convert from gambit type solutions.
    converted_solutions = []
    strategy_start_indices = np.cumsum(payoffs.shape[:-1])
    for solution in solutions:
      mixtures = np.split(solution, strategy_start_indices)[:-1]  # Remove empty slice at end.
      converted_solutions.append(dict(enumerate(mixtures)))

    return converted_solutions
