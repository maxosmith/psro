"""Test for `uniform` strategy solvers."""
import numpy as np
from absl.testing import absltest, parameterized
from marl.utils import tree_utils

from psro import strategy
from psro.solvers import uniform


class UniformTest(parameterized.TestCase):
  """Test suite for `Uniform`."""

  @parameterized.parameters(
      {"payoffs": np.zeros((2, 2, 2)), "expected": {0: np.array([0.5, 0.5]), 1: np.array([0.5, 0.5])}},
      {
          "payoffs": np.zeros((2, 3, 2)),
          "expected": {0: np.array([0.5, 0.5]), 1: np.array([0.3333333, 0.3333333, 0.3333333])},
      },
  )
  def test_uniform(self, payoffs: np.ndarray, expected: strategy.Profile):
    """Tests basic API."""
    tree_utils.assert_almost_equals(expected, uniform.Uniform()(payoffs))


class UniformBiasedTest(parameterized.TestCase):
  """Test suite for `UniformBiased`."""

  @parameterized.parameters(
      {
          "payoffs": np.zeros((2, 2, 2)),
          "expected": {0: np.array([0.268941, 0.731059]), 1: np.array([0.268941, 0.731059])},
      },
      {
          "payoffs": np.zeros((2, 3, 2)),
          "expected": {0: np.array([0.268941, 0.731059]), 1: np.array([0.090031, 0.244728, 0.665241])},
      },
  )
  def test_uniform(self, payoffs: np.ndarray, expected: strategy.Profile):
    """Tests basic API."""
    tree_utils.assert_almost_equals(expected, uniform.UniformBiased()(payoffs))


if __name__ == "__main__":
  absltest.main()
