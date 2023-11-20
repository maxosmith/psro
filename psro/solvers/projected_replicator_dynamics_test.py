"""Test for `projected_replicator_dynamics`."""
import numpy as np
from absl.testing import absltest, parameterized

from psro.solvers import projected_replicator_dynamics


class ProjectedReplicatorDynamicsTest(parameterized.TestCase):
  """Test suite for `ProjectedReplicatorDynamics`."""

  def test_two_players(self):
    """Tests standard call to PRD."""
    player_0 = np.array([[2, 1, 0], [0, -1, -2]])
    player_1 = np.array([[2, 1, 0], [0, -1, -2]])
    payoff_matrix = np.stack([player_0, player_1], axis=-1)

    solver = projected_replicator_dynamics.ProjectedReplicatorDynamics(
        num_iterations=50_000,
        dt=1e-3,
        gamma=1e-8,
        average_over_last_n_strategies=10,
    )
    solution = solver(payoff_matrix)

    self.assertLen(solution, 2)
    self.assertLen(solution[0], 2)
    self.assertLen(solution[1], 3)
    self.assertGreater(solution[0][0], 0.999)

  def test_three_players(self):
    """PRD with three players."""
    player_0 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    player_1 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    player_2 = np.array([[[2, 1, 0], [1, 0, -1]], [[1, 0, -1], [0, -1, -2]]])
    payoff_matrix = np.stack([player_0, player_1, player_2], axis=-1)

    solver = projected_replicator_dynamics.ProjectedReplicatorDynamics(
        num_iterations=50_000,
        dt=1e-3,
        gamma=1e-8,
        average_over_last_n_strategies=10,
    )
    solution = solver(payoff_matrix)

    self.assertLen(solution, 3)
    self.assertLen(solution[0], 2)
    self.assertLen(solution[1], 2)
    self.assertLen(solution[2], 3)
    self.assertGreater(solution[0][0], 0.999)


if __name__ == "__main__":
  absltest.main()
