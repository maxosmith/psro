"""Utility functions for computing best responses through OpenSpiel."""
import copy
import itertools

import numpy as np
import pyspiel
from marl import individuals, types, worlds
from marl.games import openspiel_proxy
from open_spiel.python import policy
from open_spiel.python.algorithms import get_all_states, policy_aggregator
from open_spiel.python.algorithms.psro_v2 import utils as openspiel_utils
from scipy.special import softmax

from psro import strategy

_RPS_STRING = "repeated_game(stage_game=matrix_rps(),num_repetitions=2)"


def _state_sequences_iter(game):
  """Iterate over state sequences in the game."""

  def recurse(state, state_sequence):
    """Generator for state sequences."""
    yield (state_sequence, state)
    if state.is_terminal():
      return
    elif state.is_chance_node():
      for action, _ in state.chance_outcomes():
        child_state = state.child(action)
        yield from recurse(child_state, state_sequence)
    elif state.is_simultaneous_node():
      joint_actions = []
      for player in range(game.num_players()):
        joint_actions.append(state.legal_actions(player))
      for actions in itertools.product(*joint_actions):
        child_state = state.clone()
        child_state.apply_actions(actions)
        new_state_sequence = state_sequence + [child_state.serialize()]
        yield from recurse(child_state, new_state_sequence)
    else:
      for action in state.legal_actions():
        child_state = state.child(action)
        new_state_sequence = state_sequence + [child_state.serialize()]
        yield from recurse(child_state, new_state_sequence)

  initial_state = game.new_initial_state()
  initial_state_serialized = initial_state.serialize()
  yield from recurse(initial_state, [initial_state_serialized])


def _history_to_timesteps(pygame, history, final_state) -> list[worlds.TimeStep]:
  """Convert an OpenSpiel history to a sequence of timesteps."""
  proxy = openspiel_proxy.OpenSpielProxy(pygame, include_full_state=True)
  timesteps = []
  for state in history:
    state = pygame.deserialize_state(state)
    if state.is_chance_node():
      continue
    proxy._game._state = state  # pylint: disable=protected-access
    timesteps.append(proxy._convert_openspiel_timestep(proxy._game.get_time_step()))  # pylint: disable=protected-access
  proxy._game._state = final_state  # pylint: disable=protected-access
  timesteps.append(proxy._convert_openspiel_timestep(proxy._game.get_time_step()))  # pylint: disable=protected-access
  return timesteps


def compute_states_and_info_states_if_none(game, all_states=None, state_to_information_state=None, player_id=None):
  """Returns all_states and/or state_to_information_state for the game.

  To recompute everything, pass in None for both all_states and
  state_to_information_state. Otherwise, this function will use the passed in
  values to reconstruct either of them.

  Args:
    game: The open_spiel game.
    all_states: The result of calling get_all_states.get_all_states. Cached for
      improved performance.
    state_to_information_state: A dict mapping str(state) to
      state.information_state for every state in the game. Cached for improved
      performance.
  """
  if all_states is None:
    all_states = get_all_states.get_all_states(
        game, depth_limit=-1, include_terminals=False, include_chance_states=False
    )

  if state_to_information_state is None:
    if player_id is not None:
      state_to_information_state = {
          state: all_states[state].information_state_string(player_id) for state in all_states
      }
    else:
      state_to_information_state = {state: all_states[state].information_state_string() for state in all_states}

  return all_states, state_to_information_state


def bot_to_openspiel_policy(pygame: pyspiel.Game, bot: individuals.Bot, player_id: int = 0) -> policy.TabularPolicy:
  """."""
  tab_policy = policy.TabularPolicy(pygame, players=[player_id])

  # Collect the information states that occurred leading up to each game-state.
  serial_to_state = {state.serialize(): state for state in tab_policy.states}
  serial_to_history = {}
  for sequence, state in _state_sequences_iter(pygame):
    serial = state.serialize()
    if state.serialize() in serial_to_state:
      if serial in serial_to_history:
        raise RuntimeError("Found multiple histories that lead to same game state.")
      serial_to_history[serial] = sequence

  for state_i, state in enumerate(tab_policy.states):
    # Get the sequence of timesteps leading up to the game-state.
    history = serial_to_history[state.serialize()]
    timesteps = _history_to_timesteps(pygame, history, state)

    reset = False
    agent_state = None
    action = None
    for timestep in timesteps:
      if (player_id in timestep) and not reset:
        # Play the bot through the history so it has an accurate internal state.
        agent_state = bot.episode_reset(timestep[player_id])
        reset = True
      if player_id in timestep:
        agent_state, action = bot.step(agent_state, timestep[player_id])

    if action is not None:
      if hasattr(agent_state, "logits"):
        tab_policy.action_probability_array[state_i] = softmax(agent_state.logits)
      else:
        # Convert final action into new tabular policy.
        tab_policy.action_probability_array[state_i] = np.zeros_like(tab_policy.action_probability_array[state_i])
        tab_policy.action_probability_array[state_i][action] = 1

  return tab_policy


def aggregate_joint_strategy(
    pygame: pyspiel.Game, players: strategy.JointStrategy, learner_id: None | types.PlayerID = None
) -> policy_aggregator.PolicyFunction:
  """Aggregate a strategy into a single OpenSpiel-compatible policy.

  Args:
    pygame:
    players:
    learner_id: ID for a policy that will be learned. This is specified to avoid the unnecessary conversion
      of this player's strategy into OpenSpiel policies.
  """
  current_best = []
  probabilities_of_playing_policies = []
  for pid in range(len(players)):
    if (learner_id is not None) and (learner_id == pid):
      # For the player we're going to compute the BR for we can skip converting their
      # strategy as its an expensive calculation.
      current_best.append([copy.copy(policy.TabularPolicy(pygame))])
      probabilities_of_playing_policies.append([1.0])
    else:
      current_best.append(
          [
              bot_to_openspiel_policy(pygame, players[pid].policies[policy_id], pid)
              for policy_id in range(len(players[pid].policies))
          ]
      )
      probabilities_of_playing_policies.append(list(players[pid].mixture))

  return openspiel_utils.aggregate_policies(
      pygame,
      current_best,
      probabilities_of_playing_policies,
  )
