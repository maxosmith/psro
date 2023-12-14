"""Test for `EmpiricalNFG`."""
import numpy as np
from absl.testing import absltest, parameterized

from psro.empirical_games import normal_form_sql


class EmpiricalNFGTest(parameterized.TestCase):
  """Test suite for `EmpiricalNFG`."""

  @parameterized.parameters(
      ({0: 0, 1: 1}, {0: np.array([2.5]), 1: np.array([3.5])}),
      ({0: 1, 1: 0}, {0: np.array([1.5]), 1: np.array([2.5])}),
      ({0: 1, 1: 1}, {0: np.array([0.5]), 1: np.array([1.5])}),
  )
  def test_add_and_retrieval(self, profile, payoffs):
    """Test adding and retrieving payoffs."""
    game = normal_form_sql.NormalForm(num_players=2)
    game.add_payoffs(profile, payoffs)
    self.assertIn(profile, game)
    samples = game.payoff_samples(profile)
    self.assertTrue(np.array_equal(samples[0], payoffs[0]))
    self.assertTrue(np.array_equal(samples[1], payoffs[1]))

  @parameterized.parameters(
      ({0: 0, 1: 1}, {0: np.array([2.5, 3.5]), 1: np.array([3.5, 4.5])}, {0: 3.0, 1: 4.0}),
      ({0: 1, 1: 0}, {0: np.array([1.5, 2.0]), 1: np.array([2.5, 3.5])}, {0: 1.75, 1: 3.0}),
  )
  def test_average_payoffs(self, profile, payoffs, expected_avg):
    """Test `average_payoffs`."""
    game = normal_form_sql.NormalForm(num_players=2)
    game.add_payoffs(profile, payoffs)
    avg_payoffs = game.average_payoffs(profile)
    self.assertAlmostEqual(avg_payoffs[0], expected_avg[0])
    self.assertAlmostEqual(avg_payoffs[1], expected_avg[1])

  @parameterized.parameters(
      ({0: 0, 1: 1}, {0: np.array([2.5]), 1: np.array([3.5])}, 1),
      ({0: 1, 1: 0}, {0: np.array([1.5, 2.5]), 1: np.array([2.5, 3.5, 4.5])}, 2),
  )
  def test_num_samples(self, profile, payoffs, expected_count):
    """Test `num_samples`."""
    game = normal_form_sql.NormalForm(num_players=2)
    game.add_payoffs(profile, payoffs)
    self.assertEqual(game.num_samples(profile), expected_count)

  @parameterized.parameters(
      (
          (
              ({0: 0, 1: 0}, {0: np.array([1]), 1: np.array([2])}),
              ({0: 1, 1: 0}, {0: np.array([3]), 1: np.array([4])}),
              ({0: 0, 1: 1}, {0: np.array([5]), 1: np.array([6])}),
              ({0: 1, 1: 1}, {0: np.array([7]), 1: np.array([8])}),
          ),
          np.array([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]),
      ),
      (
          (({0: 0, 1: 0}, {0: np.array([1]), 1: np.array([2])}),),
          np.array([[[1.0, 2.0]]]),
      ),
  )
  def test_game_matrix(self, payoffs_to_add, expected_matrix):
    """Test `game_matrix`."""
    game = normal_form_sql.NormalForm(num_players=2)
    for profile, payoffs in payoffs_to_add:
      game.add_payoffs(profile, payoffs)
    matrix = game.game_matrix()
    self.assertTrue(np.array_equal(matrix, expected_matrix))

  @parameterized.parameters(
      (
          (
              ({0: 0, 1: 0}, {0: np.array([1]), 1: np.array([1])}),
              ({0: 1, 1: 0}, {0: np.array([1]), 1: np.array([1])}),
              ({0: 0, 1: 1}, {0: np.array([1]), 1: np.array([1])}),
              ({0: 1, 1: 1}, {0: np.array([1]), 1: np.array([1])}),
          ),
          ({0: 0, 1: 0}, {0: 1, 1: 0}, {0: 0, 1: 1}, {0: 1, 1: 1}),
          ({0: 2, 1: 0}, {0: 2, 1: 1}, {0: 2, 1: 2}, {0: 0, 1: 2}, {0: 1, 1: 2}),
      ),
      (
          (),
          (),
          ({0: 0, 1: 0}, {0: 2, 1: 2}),
      ),
  )
  def test_contains(self, payoffs_to_add, profiles_in, profiles_not_in):
    """Test `__contains__`."""
    game = normal_form_sql.NormalForm(num_players=2)
    for profile, payoffs in payoffs_to_add:
      game.add_payoffs(profile, payoffs)
    for profile in profiles_in:
      self.assertIn(profile, game)
    for profile in profiles_not_in:
      self.assertNotIn(profile, game)


if __name__ == "__main__":
  absltest.main()
