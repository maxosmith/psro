"""Test for `openspiel_br.utils`."""
import pyspiel
from absl.testing import absltest, parameterized
from marl import bots
from marl.utils import tree_utils

from psro import strategy
from psro.response_oracles.openspiel_br import utils


class UtilsTest(parameterized.TestCase):
  """Test suite for `utils`."""

  @parameterized.parameters(
      {"game_name": "kuhn_poker", "player_id": 0, "action": 0},
      {"game_name": "kuhn_poker", "player_id": 0, "action": 1},
      {"game_name": "kuhn_poker", "player_id": 1, "action": 0},
      {"game_name": "kuhn_poker", "player_id": 1, "action": 1},
      {"game_name": "leduc_poker", "player_id": 1, "action": 1},
  )
  def test_bot_to_openspiel_policy_constant(self, game_name: str, player_id: int, action: int):
    """Tests `bot_to_openspiel_policy` with constant actions."""
    game = pyspiel.load_game(game_name)
    bot = bots.ConstantActionBot(action)
    os_policy = utils.bot_to_openspiel_policy(game, bot, player_id=player_id)

    states = set(s.information_state_string(player_id) for s in os_policy.states if not s.is_chance_node())
    dict_policy = {k: v for k, v in os_policy.to_dict().items() if k in states}

    for _, decisions in dict_policy.items():
      decisions = dict(decisions)
      for cur_action, probability in decisions.items():
        if cur_action == action:
          self.assertAlmostEqual(1, probability)
        else:
          self.assertAlmostEqual(0, probability)

  def test_aggregate_joint_policy(self):
    """Tests `aggregate_joint_policy`."""
    game = pyspiel.load_game("kuhn_poker")
    players = {
        0: strategy.Strategy(policies=[bots.ConstantActionBot(0)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.ConstantActionBot(1)], mixture=[1.0]),
    }
    agg = utils.aggregate_joint_strategy(game, players)

    tab_policy = agg.to_tabular()
    dict_policy = tab_policy.to_dict()

    for state in tab_policy.states:
      current_player = state.current_player()
      info_str = state.information_state_string(current_player)
      decisions = dict(dict_policy[info_str])

      for action, probability in decisions.items():
        if current_player == action:
          self.assertAlmostEqual(1, probability)
        else:
          self.assertAlmostEqual(0, probability)

  def test_aggregate_stochastic_policies(self):
    """Tests `aggregate_joint_policy`."""
    game = pyspiel.load_game("kuhn_poker")
    players = {
        0: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
    }
    agg = utils.aggregate_joint_strategy(game, players)

    tab_policy = agg.to_tabular()
    dict_policy = tab_policy.to_dict()

    for actions_with_probs in dict_policy.values():
      for _, prob in actions_with_probs:
        self.assertAlmostEqual(0.5, prob)

  def test_aggregate_stochastic_policies2(self):
    """Tests `aggregate_joint_policy`."""
    game = pyspiel.load_game("kuhn_poker")
    players = {
        0: strategy.Strategy(policies=[bots.RandomActionBot(2)], mixture=[1.0]),
        1: strategy.Strategy(policies=[bots.ConstantActionBot(1)], mixture=[1.0]),
    }
    agg = utils.aggregate_joint_strategy(game, players)

    tab_policy = agg.to_tabular()
    dict_policy = tab_policy.to_dict()

    for state, actions_with_probs in dict_policy.items():
      if len(state) == 2:  # Player 1.
        self.assertAlmostEquals(0.0, actions_with_probs[0][1])
        self.assertAlmostEquals(1.0, actions_with_probs[1][1])
      else:  # Player 0.
        self.assertAlmostEquals(0.5, actions_with_probs[0][1])
        self.assertAlmostEquals(0.5, actions_with_probs[1][1])

  @parameterized.parameters(
      {
          "player0_mixture": [0.8, 0.2],
          "player1_action": 1,
          "expected_policy": {
              # Player 0, start of game.
              "0": {0: 0.8, 1: 0.2},
              "1": {0: 0.8, 1: 0.2},
              "2": {0: 0.8, 1: 0.2},
              # Player 0, when Player 0 passed and then Player 1 bet.
              "0pb": {0: 1.0, 1: 0.0},
              "1pb": {0: 1.0, 1: 0.0},
              "2pb": {0: 1.0, 1: 0.0},
              # Player 1, when Player 0 passed.
              "0p": {0: 0.0, 1: 1.0},
              "1p": {0: 0.0, 1: 1.0},
              "2p": {0: 0.0, 1: 1.0},
              # Player 1, when Player 1 bet.
              "0b": {0: 0.0, 1: 1.0},
              "1b": {0: 0.0, 1: 1.0},
              "2b": {0: 0.0, 1: 1.0},
          },
      },
      {
          "player0_mixture": [0.7, 0.3],
          "player1_action": 1,
          "expected_policy": {
              "0": {0: 0.7, 1: 0.3},
              "1": {0: 0.7, 1: 0.3},
              "2": {0: 0.7, 1: 0.3},
              "0pb": {0: 1.0, 1: 0.0},
              "1pb": {0: 1.0, 1: 0.0},
              "2pb": {0: 1.0, 1: 0.0},
              "0p": {0: 0.0, 1: 1.0},
              "1p": {0: 0.0, 1: 1.0},
              "2p": {0: 0.0, 1: 1.0},
              "0b": {0: 0.0, 1: 1.0},
              "1b": {0: 0.0, 1: 1.0},
              "2b": {0: 0.0, 1: 1.0},
          },
      },
      {
          "player0_mixture": [0.7, 0.3],
          "player1_action": 0,
          "expected_policy": {
              "0": {0: 0.7, 1: 0.3},
              "1": {0: 0.7, 1: 0.3},
              "2": {0: 0.7, 1: 0.3},
              "0pb": {0: 1.0, 1: 0.0},
              "1pb": {0: 1.0, 1: 0.0},
              "2pb": {0: 1.0, 1: 0.0},
              "0p": {0: 1.0, 1: 0.0},
              "1p": {0: 1.0, 1: 0.0},
              "2p": {0: 1.0, 1: 0.0},
              "0b": {0: 1.0, 1: 0.0},
              "1b": {0: 1.0, 1: 0.0},
              "2b": {0: 1.0, 1: 0.0},
          },
      },
  )
  def test_aggregate_joint_policy_mixed_strateg(self, player0_mixture, player1_action, expected_policy):
    """Tests `aggregate_joint_policy`."""
    game = pyspiel.load_game("kuhn_poker")
    players = {
        0: strategy.Strategy(
            policies=[
                bots.ConstantActionBot(0),
                bots.ConstantActionBot(1),
            ],
            mixture=player0_mixture,
        ),
        1: strategy.Strategy(policies=[bots.ConstantActionBot(player1_action)], mixture=[1.0]),
    }
    agg = utils.aggregate_joint_strategy(game, players)
    dict_policy = {k: dict(v) for k, v in agg.to_tabular().to_dict().items()}
    tree_utils.assert_almost_equals(expected_policy, dict_policy)


if __name__ == "__main__":
  absltest.main()
