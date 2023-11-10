"""Compute an exact best response using OpenSpiel.

References:
 - https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/psro_v2/best_response_oracle.py  # pylint: disable=line-too-long
"""
import dataclasses
from typing import Tuple

import numpy as np
import pyspiel
from marl import individuals, types, worlds
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import best_response as openspiel_br
from open_spiel.python.algorithms import policy_utils as openspiel_policy_utils
from open_spiel.python.algorithms.psro_v2 import utils as openspiel_utils

from psro import core, strategy
from psro.response_oracles.openspiel_br import utils


@dataclasses.dataclass
class OpenSpielBestResponseProxy(individuals.Bot):
  """Proxy for an OpenSpiel's BestResponse policy."""

  player_id: types.PlayerID
  br: openspiel_br.CPPBestResponsePolicy | openspiel_br.BestResponsePolicy

  def step(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.State, types.Action]:
    _, game_state = pyspiel.deserialize_game_and_state(timestep.observation["serialized_state"])
    return state, self.br.best_response_action(game_state.information_state_string(self.player_id))

  def episode_reset(self, timestep: worlds.TimeStep):
    del timestep
    return ()


class BestResponse:
  """Oracle for computing exact best responses through OpenSpiel.

  Args:
  game: The game on which the optimization process takes place. Only sequential games
    are supported. Simultaneous games can be changed into sequential games through its game string:
      turn_based_simultaneous_game(game=...)
  """

  def __init__(
      self,
      game: pyspiel.Game,
      backend: str = "cpp",
      prob_cut_threshold: float = -1.0,
      action_value_tolerance: float = -1.0,
  ) -> None:
    """Initializer."""
    self._game = game
    self._backend = backend
    self._prob_cut_threshold = prob_cut_threshold
    self._action_value_tolerance = action_value_tolerance

    self._all_states = None
    self._state_to_information_state = None

  def _maybe_initialize_resources(self):
    """Lazily initialize resources used during BR computation."""
    # Should compute all_states and state_to_information_state only once in
    # the program, as caching them speeds up TabularBestResponse tremendously.
    if self._all_states is None:
      self._all_states, self._state_to_information_state = utils.compute_states_and_info_states_if_none(
          self._game, self._all_states, self._state_to_information_state
      )

    if self._backend == "cpp":
      policy = openspiel_policy.UniformRandomPolicy(self._game)
      policy_to_dict = openspiel_policy_utils.policy_to_dict(
          policy, self._game, self._all_states, self._state_to_information_state
      )

      self.best_response_processors = []
      self.best_responders = []
      for best_responder_id in range(self._game.num_players()):
        self.best_response_processors.append(
            pyspiel.TabularBestResponse(
                self._game, best_responder_id, policy_to_dict, self._prob_cut_threshold, self._action_value_tolerance
            )
        )
        self.best_responders.append(
            openspiel_br.CPPBestResponsePolicy(
                self._game,
                best_responder_id,
                policy,
                self._all_states,
                self._state_to_information_state,
                self.best_response_processors[best_responder_id],
            )
        )

  def __call__(
      self,
      job: core.ResponseOracleJob,
      **kwargs,
  ) -> OpenSpielBestResponseProxy:
    """Compute a best response.

    Args:
      job:
      **kwargs: Other set of arguments, for compatibility purposes.
    """
    del kwargs
    players = job.players
    learner_id = job.learner_id
    self._validate_joint_strategy(players)
    self._maybe_initialize_resources()

    # Convert the policies into OpenSpiel compatible policies.
    current_best = []
    probabilities_of_playing_policies = []
    for pid in range(len(players)):
      if learner_id == pid:
        # For the player we're going to compute the BR for we can skip converting their
        # strategy as its an expensive calculation.
        current_best.append([openspiel_policy.TabularPolicy(self._game).__copy__()])
        probabilities_of_playing_policies.append([1.0])
      else:
        current_best.append(
            [
                utils.bot_to_openspiel_policy(self._game, players[pid].policies[policy_id], pid)
                for policy_id in range(len(players[pid].policies))
            ]
        )
        probabilities_of_playing_policies.append(list(players[pid].mixture))

    aggr_policy = openspiel_utils.aggregate_policies(
        self._game,
        current_best,
        probabilities_of_playing_policies,
    )

    # This takes as input an aggregate policy, and computes a best response
    # for current_player at the applicable information states by recursing
    # through the game tree. At information states involving other players
    # or chance, the aggr_policy is used to compute the expected value, such
    # that a best response for current_player can be computed.
    if self._backend == "py":
      best_resp = openspiel_br.BestResponsePolicy(self._game, learner_id, aggr_policy)
    else:
      self.best_response_processors[learner_id].set_policy(
          openspiel_policy_utils.policy_to_dict(
              aggr_policy,
              self._game,
              self._all_states,
              self._state_to_information_state,
          )
      )

      self.best_responders[learner_id] = openspiel_br.CPPBestResponsePolicy(
          self._game,
          learner_id,
          aggr_policy,
          self._all_states,
          self._state_to_information_state,
          self.best_response_processors[learner_id],
      )
      best_resp = self.best_responders[learner_id]
    return (job.learner_id, OpenSpielBestResponseProxy(learner_id, best_resp))

  def _validate_joint_strategy(self, players: strategy.JointStrategy):
    """Validate that the joint strategy is valide for the game."""
    if self._game.num_players() != len(players):
      raise ValueError("Best response requires strategies for all players.")
    for pid in range(self._game.num_players()):
      if pid not in players:
        raise ValueError(f"Strategy not defined for {pid=}.")
