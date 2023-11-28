"""Test for `pyspiel_nash`."""
import numpy as np
from absl.testing import absltest, parameterized

from psro.solvers import pyspiel_nash


class PySpielNashTest(parameterized.TestCase):
  """Test suite for `pyspiel_nash.Nash`."""

  def test_rock_paper_scissors(self):
    """Tests RPS."""
    solution = pyspiel_nash.Nash()(
        np.swapaxes(
            np.array([
                [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
                [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
            ]),
            0,
            -1,
        )
    )[0]
    self.assertLen(solution, 2)
    self.assertLen(solution[0], 3)
    self.assertLen(solution[1], 3)

    for i in range(3):
      self.assertAlmostEqual(solution[0][i], 1.0 / 3.0)
      self.assertAlmostEqual(solution[1][i], 1.0 / 3.0)

  def test_biased_rock_paper_scissors(self):
    """Tests Biased-RPS."""
    solution = pyspiel_nash.Nash()(
        np.swapaxes(
            np.array([
                [[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],
                [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]],
            ]),
            0,
            -1,
        )
    )[0]
    self.assertLen(solution, 2)
    self.assertLen(solution[0], 3)
    self.assertLen(solution[1], 3)

    for i in range(2):
      self.assertAlmostEqual(solution[i][0], 1.0 / 16.0, places=4)
      self.assertAlmostEqual(solution[i][1], 10.0 / 16.0, places=4)
      self.assertAlmostEqual(solution[i][2], 5.0 / 16.0, places=4)


if __name__ == "__main__":
  absltest.main()
