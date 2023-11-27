"""Test for `regret_matching`."""
import numpy as np
import pyspiel
from absl.testing import absltest, parameterized
from open_spiel.python.egt.utils import game_payoffs_array

from psro.solvers import regret_matching


class RegretMatchingTest(parameterized.TestCase):
  """Test suite for `regret_matching`."""

  def test_two_players(self):
    """Tests two-player games."""
    player_0 = np.array([[2, 1, 0], [0, -1, -2]])
    player_1 = np.array([[2, 1, 0], [0, -1, -2]])
    payoff_matrix = np.stack([player_0, player_1], axis=-1)

    solver = regret_matching.RegretMatching(iterations=50_000, gamma=1e-8, average_over_last_n_strategies=10)
    solution = solver(payoff_matrix)
    self.assertLen(solution, 1)
    solution = solution[0]

    self.assertLen(solution, 2)
    self.assertLen(solution[0], 2)
    self.assertLen(solution[1], 3)
    self.assertGreater(solution[0][0], 0.999)
    self.assertGreater(solution[1][0], 0.999)

  def test_three_players(self):
    """Tests three-player games."""
    player_0 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    player_1 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    player_2 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    payoff_matrix = np.stack([player_0, player_1, player_2], axis=-1)

    solver = regret_matching.RegretMatching(iterations=50_000, gamma=1e-6, average_over_last_n_strategies=10)
    solution = solver(payoff_matrix)[0]

    self.assertLen(solution, 3, "Wrong strategy length.")
    self.assertGreater(solution[0][0], 0.999, "Regret matching failed in trivial case.")
    self.assertGreater(solution[1][0], 0.999, "Regret matching failed in trivial case.")
    self.assertGreater(solution[2][0], 0.999, "Regret matching failed in trivial case.")

  def test_biased_rps(self):
    """Tests a biased instance of RPS."""
    game = pyspiel.load_game("matrix_brps")
    payoff_matrix = game_payoffs_array(game)
    payoff_matrix = np.moveaxis(payoff_matrix, 0, -1)

    solver = regret_matching.RegretMatching(iterations=50_000, gamma=1e-6)
    solution = solver(payoff_matrix)[0]

    self.assertLen(solution, 2, "Wrong strategy length.")
    # places=1 corresponds to an absolute difference of < 0.01
    self.assertAlmostEqual(solution[0][0], 1 / 16.0, places=1)
    self.assertAlmostEqual(solution[0][1], 10 / 16.0, places=1)
    self.assertAlmostEqual(solution[0][2], 5 / 16.0, places=1)


if __name__ == "__main__":
  absltest.main()
