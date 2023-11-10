"""Test for `TODO`."""
from typing import Mapping

import numpy as np
from absl.testing import absltest, parameterized
from marl import types

from psro import strategy
from psro.utils import regret_utils


class TODOTest(parameterized.TestCase):
  """Test suite for `TODO`."""

  @parameterized.parameters(
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [1, 0], 1: [1, 0]},
          "payoffs": {0: 0.30, 1: -0.30},
      },
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [0, 1], 1: [0, 1]},
          "payoffs": {0: 0.23, 1: -0.23},
      },
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [0.2, 0.8], 1: [0.4, 0.6]},
          "payoffs": {0: 0.2816, 1: -0.2816},
      },
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80], [-0.16, 0.16]],
              [[0.76, -0.76], [0.23, -0.23], [-0.20, 0.20]],
              [[0.16, -0.16], [0.53, -0.53], [-0.43, 0.43]],
              [[0.16, -0.16], [0.53, -0.53], [-0.43, 0.43]],
          ]),
          "solution": {0: [0.5, 0.5, 0, 0], 1: [0, 1, 0]},
          "payoffs": {0: -0.285, 1: 0.285},
      },
  )
  def test_solution_payoffs(
      self, game_matrix: np.ndarray, solution: strategy.JointStrategy, payoffs: Mapping[types.PlayerID, float]
  ):
    """Test `solution_payoffs`."""
    result = regret_utils.solution_payoffs(game_matrix, solution)
    for key, value in payoffs.items():
      self.assertAlmostEqual(value, result[key])

  @parameterized.parameters(
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [1, 0]},
          "collapsed": np.array([[0.30, -0.30], [-0.8, 0.80]]),
      },
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {1: [1, 0]},
          "collapsed": np.array([[0.3, -0.3], [0.76, -0.76]]),
      },
      {
          "game_matrix": np.array([
              [[[1, -1, 0], [2, -2, 1]], [[-1, 1, -1], [-2, 2, 0]]],
              [[[1, -1, 1], [2, -2, -1]], [[-1, 1, 0], [-2, 2, 1]]],
          ]),
          "solution": {1: [1, 0], 2: [0, 1]},
          "collapsed": np.array([[2, -2, 1], [2, -2, -1]]),
      },
      {
          "game_matrix": np.array([
              [[[1, -1, 0], [2, -2, 1]], [[-1, 1, -1], [-2, 2, 0]]],
              [[[1, -1, 1], [2, -2, -1]], [[-1, 1, 0], [-2, 2, 1]]],
          ]),
          "solution": {2: [0.5, 0.5]},
          "collapsed": np.array([[[1.5, -1.5, 0.5], [-1.5, 1.5, -0.5]], [[1.5, -1.5, 0.0], [-1.5, 1.5, 0.5]]]),
      },
      {
          "game_matrix": np.array([
              [[[1, -1, 0], [2, -2, 1]], [[-1, 1, -1], [-2, 2, 0]]],
              [[[1, -1, 1], [2, -2, -1]], [[-1, 1, 0], [-2, 2, 1]]],
          ]),
          "solution": {1: [0.3, 0.7], 2: [0.5, 0.5]},
          "collapsed": np.array([[-0.6, 0.6, -0.2], [-0.6, 0.6, 0.35]]),
      },
      {
          "game_matrix": np.array([
              [[[1, -1, 0], [2, -2, 1]], [[-1, 1, -1], [-2, 2, 0]]],
              [[[1, -1, 1], [2, -2, -1]], [[-1, 1, 0], [-2, 2, 1]]],
          ]),
          "solution": {1: [0.3, 0.7]},
          "collapsed": np.array([[[-0.4, 0.4, -0.7], [-0.8, 0.8, 0.3]], [[-0.4, 0.4, 0.3], [-0.8, 0.8, 0.4]]]),
      },
  )
  def test_reduce_payoffs(self, game_matrix: np.ndarray, solution: strategy.JointStrategy, collapsed: np.ndarray):
    """Test `reduce_payoffs`."""
    print(regret_utils.reduce_payoffs(game_matrix, solution))
    np.testing.assert_almost_equal(collapsed, regret_utils.reduce_payoffs(game_matrix, solution))

  @parameterized.parameters(
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [1, 0], 1: [1, 0]},
          "regret": {0: 0.46, 1: 1.1},
      },
      {
          "game_matrix": np.array([
              [[0.30, -0.30], [-0.8, 0.80]],
              [[0.76, -0.76], [0.23, -0.23]],
          ]),
          "solution": {0: [0, 1], 1: [1, 0]},
          "regret": {0: 0.0, 1: 0.53},
      },
  )
  def test_regret(
      self, game_matrix: np.ndarray, solution: strategy.JointStrategy, regret: Mapping[types.PlayerID, float]
  ):
    """Test `regret`."""
    result = regret_utils.regret(game_matrix, solution)
    for key, value in regret.items():
      self.assertAlmostEqual(value, result[key])


if __name__ == "__main__":
  absltest.main()
