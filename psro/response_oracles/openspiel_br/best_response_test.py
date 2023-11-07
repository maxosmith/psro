"""Test for `openspiel_br`."""
import pyspiel
from absl.testing import absltest, parameterized
from marl import bots

from psro import strategy
from psro.response_oracles.openspiel_br import best_response


class BestResponseTest(parameterized.TestCase):
  """Test suite for `TODO`."""

  @parameterized.parameters(
      {"game_name": "kuhn_poker", "num_actions": 2},
      {"game_name": "misere(game=kuhn_poker())", "num_actions": 2},
      {"game_name": "hex(board_size=2)", "num_actions": 4},
      {"game_name": "go(board_size=2)", "num_actions": 4 + 1},
      {"game_name": "leduc_poker", "num_actions": 3},
      {"game_name": "tic_tac_toe", "num_actions": 9},
  )
  def test_best_response_regression(self, game_name: str, num_actions: int):
    game = pyspiel.load_game(game_name)
    test_policy = {
        0: strategy.Strategy(policies=[bots.RandomActionBot(num_actions)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.RandomActionBot(num_actions)], mixture=[1.0]),
    }
    br_oracle = best_response.BestResponse(game, 0)
    br_oracle(players=test_policy)

  def test_best_response_behavior(self):
    game = pyspiel.load_game("kuhn_poker")
    test_policy = {
        0: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
    }
    br_oracle = best_response.BestResponse(game, 0)
    br = br_oracle(players=test_policy)
    expected_policy = {
        "0": 1,  # Bet in case opponent folds when winning
        "1": 1,  # Bet in case opponent folds when winning
        "2": 0,  # Both equally good (we return the lowest action)
        # Some of these will never happen under the best-response policy,
        # but we have computed best-response actions anyway.
        "0pb": 0,  # Fold - we're losing
        "1pb": 1,  # Call - we're 50-50
        "2pb": 1,  # Call - we've won
    }
    self.assertEqual(expected_policy, {k: br.br.best_response_action(k) for k, _ in expected_policy.items()})


if __name__ == "__main__":
  absltest.main()
