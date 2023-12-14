"""Wrappers between BSuite and Gynasmium environments."""
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from marl import specs, types, worlds


class ToGymnasium(gym.Env):
  """Convert an DeepMind environment into a Gymnasium environment."""

  # Set this in SOME subclasses
  metadata: dict[str, Any] = {"render_modes": []}
  render_mode: str | None = None
  spec: EnvSpec | None = None

  def __init__(self, env: worlds.Environment):
    """Initializer."""
    self._env = env
    self.action_space = spec_to_space(self._env.action_spec())

    observation_spec = self._env.observation_spec()
    if isinstance(observation_spec, dict) and ("info_state" in observation_spec):
      # This is a OpenSpiel game, and we need to only return "info_state" for the obseration.
      self.observation_space = spec_to_space(observation_spec["info_state"])
    else:
      self.observation_space = spec_to_space(observation_spec)

  def step(self, action: types.Action) -> tuple[types.Observation, SupportsFloat, bool, bool, dict[str, Any]]:
    """Run one timestep of the environment's dynamics using the agent's actions."""
    timestep = self._env.step(action)
    observation = None
    if isinstance(timestep.observation, dict) and ("info_state" in timestep.observation):
      observation = timestep.observation["info_state"]
    else:
      observation = timestep.observation

    return observation, timestep.reward, timestep.last(), False, {}

  def reset(
      self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[types.Observation, dict[str, Any]]:
    """Resets the enviornment to an initial internal state, returning an initial observation and info."""
    del seed, options
    timestep = self._env.reset()
    if isinstance(timestep.observation, dict) and ("info_state" in timestep.observation):
      return timestep.observation["info_state"], {}
    else:
      return timestep.observation, {}


class FromGymnasium(worlds.Environment):
  """Convert an Gymnasium environment into a DeepMind environment."""

  def __init__(self, env: gym.Env):
    """Initializer."""
    self._env = env

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    observation, reward, terminated, truncated, info = self._env.step(action)
    del info
    return worlds.TimeStep(
        step_type=worlds.StepType.LAST if terminated or truncated else worlds.StepType.MID,
        reward=reward,
        observation=observation,
    )

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    observation, info = self._env.reset()
    del info
    return worlds.TimeStep(
        step_type=worlds.StepType.FIRST,
        reward=0,
        observation=observation,
    )

  def reward_spec(self) -> specs.TreeSpec:
    """Describes the reward returned by the environment.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.ArraySpec(shape=(), dtype=float, name="reward")

  def observation_spec(self) -> specs.TreeSpec:
    """Defines the observations provided by the environment.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return space_to_spec(self._env.observation_space)

  def action_spec(self) -> specs.TreeSpec:
    """Defines the actions that should be provided to `step`.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return space_to_spec(self._env.action_space)


def spec_to_space(spec):
  """Converts a bsuite space to a gymnasium space.

  Args:
    spec: The bsuite space to be converted.

  Returns:
    A gymnasium space corresponding to the bsuite space.
  """
  if isinstance(spec, specs.DiscreteArraySpec):
    return gym.spaces.Discrete(spec.num_values)

  elif isinstance(spec, specs.BoundedArraySpec):
    return gym.spaces.Box(
        np.full(spec.shape, np.broadcast_to(spec.minimum, spec.shape)),
        np.full(spec.shape, np.broadcast_to(spec.maximum, spec.shape)),
    )

  elif isinstance(spec, specs.ArraySpec):
    return gym.spaces.Box(
        np.full(spec.shape, _np_info(spec.dtype).min),
        np.full(spec.shape, _np_info(spec.dtype).max),
    )

  else:
    raise NotImplementedError(f"Conversion for space type {type(spec)} not implemented.")


def space_to_spec(space):
  """Converts a gymnasium space to a bsuite spec.

  Args:
    space: The gymnasium space to be converted.

  Returns:
    A bsuite space corresponding to the gymnasium space.
  """
  if isinstance(space, gym.spaces.Box):
    # Check if the space is bounded in both directions
    if np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high)):
      # Bounded in both directions
      return specs.BoundedArraySpec(shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high)
    else:
      # Unbounded or partially bounded
      return specs.ArraySpec(shape=space.shape, dtype=space.dtype)

  elif isinstance(space, gym.spaces.Discrete):
    return specs.DiscreteArraySpec(num_values=space.n, dtype=int)

  else:
    raise NotImplementedError(f"Conversion for space type {type(space)} not implemented.")


def _np_info(dtype):
  """Get the info for either integer or floating types."""
  if np.issubdtype(dtype, np.integer):
    return np.iinfo(dtype)
  elif np.issubdtype(dtype, np.floating):
    return np.finfo(dtype)
  else:
    raise ValueError("Data type not supported")
