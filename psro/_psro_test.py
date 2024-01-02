"""Test for `PSRO`."""
import os
import os.path as osp
import pathlib
import tempfile

import cloudpickle
import numpy as np
import sox
from absl.testing import absltest, parameterized
from marl import bots
from marl.utils import tree_utils

from psro import _psro, empirical_games, strategy, test_utils

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
      self.assertTrue(osp.exists(tmp_dir / "empirical_game.sql"))
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


class LoadPartialRunTest(parameterized.TestCase):

  def test_reload(self):
    """Test reloading a run of PSRO."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_dir = pathlib.Path(tmp_dir)
      prev_dir = tmp_dir / "previous/"
      cont_dir = tmp_dir / "continue/"
      os.mkdir(prev_dir)
      os.mkdir(cont_dir)

      # Generate two epochs of PSRO results to load.
      eg = empirical_games.NormalFormSQL(2, prev_dir / "empirical_game.sql")
      eg.add_payoffs({0: 0, 1: 0}, {0: [0, 1, 2], 1: [3, 4, 5]})

      os.mkdir(prev_dir / "initial_strategies")
      for player in range(2):
        player_dir = prev_dir / "initial_strategies" / f"player_{player}"
        os.mkdir(player_dir)
        with open(player_dir / "policy_0.pb", "wb") as file:
          cloudpickle.dump(0, file)

      for epoch in range(3):
        epoch_dir = prev_dir / f"epoch_{epoch}"
        os.mkdir(epoch_dir)

        for player in range(2):
          player_dir = epoch_dir / f"player_{player}"
          os.mkdir(player_dir)
          with open(player_dir / "policy.pb", "wb") as file:
            cloudpickle.dump(epoch + 2, file)

        with open(epoch_dir / "game_matrix.pb", "wb") as file:
          cloudpickle.dump(sox.array_utils.uniform_like(np.zeros([epoch, epoch, 2])), file)
        with open(epoch_dir / "solution.pb", "wb") as file:
          cloudpickle.dump({p: sox.array_utils.uniform_like(np.zeros([epoch])) for p in range(2)}, file)

      os.mkdir(prev_dir / "policies")
      for player in range(2):
        player_dir = prev_dir / "policies" / f"player_{player}"
        os.mkdir(player_dir)
        for policy in range(5):
          with open(player_dir / f"policy_{policy}.pb", "wb") as file:
            cloudpickle.dump(policy, file)

      # Load the previous run.
      reloaded_psro, epoch = _psro.load_partial_run(
          game_ctor=test_utils.DummyGame,
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
              expected_games=[],
              solutions=[],
          ),
          load_dir=prev_dir,
          continue_dir=cont_dir,
      )

      self.assertEqual(epoch, 2)
      # pylint: disable=protected-access
      self.assertLen(reloaded_psro._strategies, 2)
      np.testing.assert_array_equal(reloaded_psro._strategies[0].policies, np.arange(5))
      np.testing.assert_array_equal(reloaded_psro._strategies[1].policies, np.arange(5))
      payoffs = reloaded_psro._empirical_game.payoff_samples({0: 0, 1: 0})
      self.assertLen(payoffs, 2)
      np.testing.assert_array_equal(payoffs[0], [0, 1, 2])
      np.testing.assert_array_equal(payoffs[1], [3, 4, 5])
      # pylint: enable=protected-access


if __name__ == "__main__":
  absltest.main()
