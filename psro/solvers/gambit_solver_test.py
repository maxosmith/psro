"""Test for `TODO`."""
import chex
import numpy as np
from absl.testing import absltest, parameterized

from psro.solvers import gambit_solver


class GambitSolverTest(parameterized.TestCase):
  """Test suite for `gambit_solver`."""

  @parameterized.parameters(
      (np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),),
      (np.array([[[[1, 0, 2], [0, 1, 1]], [[0, 1, 0], [1, 0, 3]]]]),),
  )
  def test_payoff_matrix_to_gambit_game(self, payoffs: np.ndarray) -> None:
    """Tests converting payoff matrix to gambit game."""
    game = gambit_solver.payoff_matrix_to_gambit_game(payoffs)
    self.assertEqual(len(game.players), len(payoffs.shape) - 1)
    self.assertEqual(len(game.strategies), np.sum(payoffs.shape[:-1]))

  @parameterized.parameters(
      (np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]), True),
      (np.array([[[1, 0], [0, 1], [1, 2]], [[0, 1], [1, 0], [1, 3]]]), True),
      (np.array([[[1, 0, 2], [0, 1, 1]], [[0, 1, 0], [1, 0, 3]]]), False),
      (np.array([[[[1, 0, 2], [0, 1, 1]], [[0, 1, 0], [1, 0, 3]]]]), True),
      (np.array([1, 0, 2]), False),
      (np.array([]), False),
  )
  def test_assert_valid_payoff_matrix(self, payoffs: np.ndarray, valid: bool) -> None:
    """Tests the validation of payoff matrices."""
    if valid:
      gambit_solver.assert_valid_payoff_matrix(payoffs)
    else:
      with self.assertRaises(ValueError):
        gambit_solver.assert_valid_payoff_matrix(payoffs)

  @parameterized.parameters(
      chex.params_product(
          (
              # Prisoner's Dilemma.
              ([[[-6, -6], [0, -10]], [[-10, 0], [-2, -2]]], [{0: [1.0, 0.0], 1: [1.0, 0.0]}]),
              # Coordination Game.
              (
                  [[[2, 2], [0, 0]], [[0, 0], [1, 1]]],
                  [
                      {0: [1.0, 0.0], 1: [1.0, 0.0]},
                      {0: [1.0 / 3.0, 2.0 / 3.0], 1: [1.0 / 3.0, 2.0 / 3.0]},
                      {0: [0.0, 1.0], 1: [0.0, 1.0]},
                  ],
              ),
              # Matching Pennies.
              ([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], [{0: [0.5, 0.5], 1: [0.5, 0.5]}]),
          ),
          ((gambit_solver.NashSolver.LCP,),),
      )
  )
  def test_gambit_solver(self, payoffs, expected_solutions, algorithm):
    """Test `GambitSolver`."""
    solver = gambit_solver.GambitSolver(algorithm)
    solutions = solver(np.array(payoffs))
    for solution, expected_solution in zip(solutions, expected_solutions):
      for key, value in expected_solution.items():
        np.testing.assert_almost_equal(solution[key], value)


if __name__ == "__main__":
  absltest.main()
