"""Wrapper that changes a multiagent interfnce (game) to single-agent (env)."""

import dataclasses

import numpy as np
from marl import specs, types, worlds

from psro import strategy


@dataclasses.dataclass
class GameToEnv(worlds.Environment):
  """Closes a game and other-players into a single-agent environment interface."""

  game: worlds.Game
  player_id: types.PlayerID
  other_players: strategy.JointStrategy | None = None

  fields = ("game", "player_id", "other_players", "_timesteps", "_player_states", "_actions")

  def __post_init__(self):
    """Post initializer."""
    if self.other_players is not None:
      assert self.player_id not in self.other_players, "Ego-player cannot also exist in other-players."
    self._timesteps = None
    self._player_states = None
    self._actions = None

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    self._timesteps = self.game.reset()
    self._player_states = {}
    self._interaction_loop()
    return self._timesteps[self.player_id]

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    self._actions[self.player_id] = action
    self._timesteps = self.game.step(self._actions)
    self._interaction_loop()
    return self._timesteps[self.player_id]

  def reward_spec(self) -> specs.TreeSpec:
    """Describes the reward returned by the environment."""
    return self.game.action_specs()[self.player_id]

  def observation_spec(self) -> specs.TreeSpec:
    """Defines the observations provided by the environment."""
    return self.game.observation_specs()[self.player_id]

  def action_spec(self) -> specs.TreeSpec:
    """Defines the actions that should be provided to `step`."""
    return self.game.action_specs()[self.player_id]

  def _interaction_loop(self):
    """Run the otheer-player and game interaction loop."""
    while True:
      # If the episode is over no interaction.
      done = [ts.last() for ts in self._timesteps.values()]
      if np.all(done):
        return

      # Collect actions from all requested players.
      self._actions = {}
      for other_player_id, timestep in self._timesteps.items():
        # Skip the ego-player because we cannot directly query it.
        if other_player_id == self.player_id:
          continue
        else:
          if other_player_id not in self._player_states:
            self._player_states[other_player_id] = self.other_players[other_player_id].episode_reset(timestep)
          self._player_states[other_player_id], self._actions[other_player_id] = self.other_players[
              other_player_id
          ].step(
              self._player_states[other_player_id],
              timestep,
          )

      # If we need the ego-player, end interaction and wait for action to be submitted.
      if self.player_id in self._timesteps:
        break

      self._timesteps = self.game.step(self._actions)


@dataclasses.dataclass
class EnvToGame(worlds.Game):
  """Convert an environment into a single-player game."""

  env: worlds.Environment

  def reset(self) -> worlds.PlayerIDToTimestep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    return {0: self.env.reset()}

  def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    return {0: self.env.step(actions[0])}

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the game to each player.p"""
    return {0: self.env.reward_spec()}

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Defines the observations provided by the game to each player."""
    return {0: self.env.observation_spec()}

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Defines the actions that should be provided to `step` by each player."""
    return {0: self.env.action_spec()}
