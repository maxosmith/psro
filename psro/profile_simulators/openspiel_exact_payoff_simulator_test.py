"""Test suite for `payoffs`."""
import pyspiel
from absl.testing import absltest, parameterized
from marl import bots
from marl.utils import tree_utils

from psro import strategy
from psro.response_oracles.openspiel_br import payoffs


class PayoffsTest(parameterized.TestCase):
  """Test suite for `payoffs`."""

  def test_payoffs_kuhn_random(self):
    """Tests `payoffs` on Kuhn Poker."""
    game = pyspiel.load_game("kuhn_poker")
    players = {
        0: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
    }
    regret = payoffs.payoffs(game, players)
    tree_utils.assert_almost_equals({0: 1 / 8, 1: -1 / 8}, regret)


if __name__ == "__main__":
  absltest.main()
