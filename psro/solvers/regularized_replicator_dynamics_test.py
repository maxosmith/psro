"""Test for `regularized_replicator_dynamics`."""
import numpy as np
from absl.testing import absltest, parameterized

from psro.solvers import regularized_replicator_dynamics


class RegularizedReplicatorDynamicsTest(parameterized.TestCase):
  """Test suite for `RegularizedReplicatorDynamics`."""

  @parameterized.parameters((False, None), (True, None), (True, 3))
  def test_averaging(self, average_strategies, window_length):
    """Tests standard call to PRD."""
    player_0 = np.array([[2, 1, 0], [0, -1, -2]])
    player_1 = np.array([[2, 1, 0], [0, -1, -2]])
    payoff_matrix = np.stack([player_0, player_1], axis=-1)

    solver = regularized_replicator_dynamics.RegularizedReplicatorDynamics(
        min_iterations=10_000,
        max_iterations=20_000,
        dt=1e-3,
        gamma=1e-8,
        average_strategies=average_strategies,
        average_strategies_window_length=window_length,
    )
    solution = solver(payoff_matrix)
    self.assertLen(solution, 1)
    solution = solution[0]

    self.assertLen(solution, 2)
    self.assertLen(solution[0], 2)
    self.assertLen(solution[1], 3)


class ComputeRegretTest(parameterized.TestCase):
  """Test suite for `compute_regret`."""

  def test_simple_case(self):
    """Tests the function with a simple case."""
    player_0_strategy = np.array([0.5, 0.5])
    player_1_strategy = np.array([0.6, 0.4])
    payoff_tensor = np.array([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]])

    expected_regrets = np.array([0.0, 1.2])
    calculated_regrets = regularized_replicator_dynamics.compute_regret(
        payoff_tensor, [player_0_strategy, player_1_strategy]
    )
    np.testing.assert_array_almost_equal(calculated_regrets, expected_regrets)

  def test_dimension_mismatch_error(self):
    """Tests if the function raises a ValueError for dimension mismatch."""
    player_0_strategy = np.array([0.5, 0.5, 0.5])
    player_1_strategy = np.array([0.6, 0.4])
    payoff_tensor = np.array([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]])

    with self.assertRaises(ValueError):
      regularized_replicator_dynamics.compute_regret(payoff_tensor, [player_0_strategy, player_1_strategy])

  @parameterized.parameters(
      (np.array([0.7, 0.3]), np.array([0.4, 0.6]), np.array([0.2, 0.8])),
      (np.array([0.3, 0.7]), np.array([0.5, 0.5]), np.array([0.1, 0.9])),
  )
  def test_different_strategies(self, player_0_strategy, player_1_strategy, player_2_strategy):
    """Tests the function with different strategies."""
    payoff_tensor = np.random.rand(2, 2, 2, 3)  # Random 3x3x3 tensor for payoff
    regrets = regularized_replicator_dynamics.compute_regret(
        payoff_tensor, [player_0_strategy, player_1_strategy, player_2_strategy]
    )
    self.assertLen(regrets, 3)  # Ensure we have 3 regrets (one per player)


if __name__ == "__main__":
  absltest.main()
