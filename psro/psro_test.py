"""Test for `PSRO`."""
import os.path as osp
import pathlib
import tempfile

import cloudpickle
import numpy as np
from absl.testing import absltest, parameterized
from marl import bots
from marl.utils import tree_utils

from psro import _psro, strategy, test_utils

_RPS = np.array([
    [[0, 0], [-1, 1], [1, -1]],
    [[1, -1], [0, 0], [-1, 1]],
    [[-1, 1], [1, -1], [0, 0]],
])


class PSROTest(parameterized.TestCase):
  """Test suite for `PSRO`."""

  def test_regression(self):
    """Regression test of PSRO."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir = pathlib.Path(tmp_dir)
      expected_solutions = [
          {0: [1.0], 1: [1.0]},
          {0: [0.0, 1.0], 1: [0.0, 1.0]},
          {0: [0.0, 0.0, 1.0], 1: [0.0, 0.0, 1.0]},
      ]
      expected_games = [_RPS[:1, :1], _RPS[:2, :2], _RPS]
      psro = _psro.PSRO(
          game_ctor=test_utils.DummyGame,
          initial_strategies={
              0: strategy.Strategy([bots.ConstantActionBot(0)], [1.0]),
              1: strategy.Strategy([bots.ConstantActionBot(1)], [1.0]),
          },
          response_oracles={
              0: test_utils.StubResponseOracle(
                  outputs=[
                      bots.ConstantActionBot(2),
                      bots.ConstantActionBot(3),
                  ],
              ),
              1: test_utils.StubResponseOracle(
                  outputs=[
                      bots.ConstantActionBot(2),
                      bots.ConstantActionBot(3),
                  ],
              ),
          },
          profile_simulator=test_utils.StubProfileSimulator(_RPS),
          game_solver=test_utils.MockGameSolver(
              expected_games=expected_games,
              solutions=expected_solutions,
          ),
          result_dir=tmp_dir,
      )
      psro.run(num_epochs=2)

      # Check all artifacts saved to disk.
      for epoch in range(1, 3):
        self.assertTrue(osp.exists(tmp_dir / f"epoch_{epoch}"))
        with open(tmp_dir / f"epoch_{epoch}" / "solution.pb", "rb") as file:
          solution = cloudpickle.load(file)
          tree_utils.assert_equals(solution, expected_solutions[epoch - 1])
        with open(tmp_dir / f"epoch_{epoch}" / "game_matrix.pb", "rb") as file:
          game = cloudpickle.load(file)
          np.testing.assert_equal(game, expected_games[epoch - 1])

      # Final results are saved at root of result directory.
      self.assertTrue(osp.exists(tmp_dir / "empirical_game.db"))
      with open(tmp_dir / "solution.pb", "rb") as file:
        solution = cloudpickle.load(file)
        tree_utils.assert_equals(solution, expected_solutions[-1])
      with open(tmp_dir / "game_matrix.pb", "rb") as file:
        game = cloudpickle.load(file)
        np.testing.assert_equal(game, expected_games[-1])

      # Check PSRO's internal state.
      np.testing.assert_equal(psro.empirical_game.game_matrix(), _RPS)
      for player in range(2):
        self.assertIn(player, psro.strategies)
        self.assertLen(psro.strategies[player], 3)


if __name__ == "__main__":
  absltest.main()
