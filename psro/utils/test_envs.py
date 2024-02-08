"""A collection of environments for testing RL algorithms."""
import random

from marl import specs, types, worlds


class RewardEnv(worlds.Environment):
  """Used to test a value learning.

  This environment has a single transition ending in a reward. This environment
  is meant to test in isolation the simplest case of learning an immediate reward:
                          V([1, 0]) = 1.
  If the agent cannot learn this then there's likely a problem with the value loss
  calculation or the optimizer.
  """

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    return worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=[1, 0])

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    del action
    return worlds.TimeStep(step_type=worlds.StepType.LAST, reward=1, observation=[0, 1])

  def reward_spec(self) -> specs.DiscreteArraySpec:
    """Describes the reward returned by the env."""
    return specs.DiscreteArraySpec(num_values=2)

  def observation_spec(self) -> specs.BoundedArraySpec:
    """Defines the observations provided by the env."""
    return specs.BoundedArraySpec(shape=(2,), dtype=int, minimum=0, maximum=1)

  def action_spec(self) -> specs.DiscreteArraySpec:
    """Defines the actions that should be provided to `step`."""
    return specs.DiscreteArraySpec(num_values=1)


class ActionValueEnv(worlds.Environment):
  """Used to test action value learning.

  This environment has two possible transitions that depend only on the action taken.
  This environment is meant to test the learner's ability to associate action with value:
                            Q([1, 0, 0], 0) = 0
                            Q([1, 0, 0], 1) = 1.
  If the agent cannot learn this then there's likely a problem with: the advantage calculation,
  policy loss, or policy update.
  """

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    return worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=[1, 0, 0])

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    if action:
      return worlds.TimeStep(step_type=worlds.StepType.LAST, reward=1, observation=[0, 1, 0])
    else:
      return worlds.TimeStep(step_type=worlds.StepType.LAST, reward=0, observation=[0, 0, 1])

  def reward_spec(self) -> specs.DiscreteArraySpec:
    """Describes the reward returned by the env."""
    return specs.DiscreteArraySpec(num_values=2)

  def observation_spec(self) -> specs.BoundedArraySpec:
    """Defines the observations provided by the env."""
    return specs.BoundedArraySpec(shape=(3,), dtype=int, minimum=0, maximum=1)

  def action_spec(self) -> specs.DiscreteArraySpec:
    """Defines the actions that should be provided to `step`."""
    return specs.DiscreteArraySpec(num_values=2)


class StateValueEnv(worlds.Environment):
  """Used to test state value learning.

  This environment tests learning values that depending on state. The episode begins
  in one of two random states, and the agent must be able to distinguish which is rewarding:
      V([1, 0, 0]) = 0                V([0, 1, 0]) = 1.
  If the agent can learn in `ValueEnv` and not in this environment this means that it
  can learn a constant reward but not a predictable reward. Therefore, there's likely
  an issue with backpropagation through the value function.
  """

  def __init__(self, seed: int | None = None):
    """Initializer.

    Args:
      seed: Random number generator's initial seed.
    """
    self.rng = random.Random(seed)
    self._reward = None

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    if self.rng.randint(0, 2):
      observation = [1, 0, 0]
      self._reward = 0
    else:
      observation = [0, 1, 0]
      self._reward = 1
    return worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=observation)

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    del action
    return worlds.TimeStep(step_type=worlds.StepType.LAST, reward=self._reward, observation=[0, 0, 1])

  def reward_spec(self) -> specs.DiscreteArraySpec:
    """Describes the reward returned by the env."""
    return specs.DiscreteArraySpec(num_values=2)

  def observation_spec(self) -> specs.BoundedArraySpec:
    """Defines the observations provided by the env."""
    return specs.BoundedArraySpec(shape=(3,), dtype=int, minimum=0, maximum=1)

  def action_spec(self) -> specs.DiscreteArraySpec:
    """Defines the actions that should be provided to `step`."""
    return specs.DiscreteArraySpec(num_values=1)


class StateActionValueEnv(worlds.Environment):
  """Used to test state-action value learning.

  This environment tests learning values that depending on both state and action. The
  episode begins in one of two random states, each with a differing rewarding action:
      Q([1, 0, 0], 0) = 1             Q([0, 1, 0], 0) = 0
      Q([1, 0, 0], 1) = 0             Q([0, 1, 0], 1) = 1.
  If `ValueEnv`, `ActionValueEnv`, and `StateValueEnv` have all been successful, but
  this environment fails, there is likely an issue with your data processing
  (e.g., stale experience or batching).
  """

  def __init__(self, seed: int | None = None):
    """Initializer.

    Args:
      seed: Random number generator's initial seed.
    """
    self.rng = random.Random(seed)
    self._observation = None

  def reset(self) -> worlds.TimeStep:
    """
    Starts a new sequence and returns the first `TimeStep` of this sequence.
    """
    self._observation = self.rng.choice([0, 1])
    observation = [0, 1, 0] if self._observation else [1, 0, 0]
    return worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=observation)

  def step(self, action: types.Action) -> worlds.TimeStep:
    """
    Updates the environment according to the action and returns a `TimeStep`.
    """
    return worlds.TimeStep(
        step_type=worlds.StepType.LAST, reward=int(action == self._observation), observation=[0, 0, 1]
    )

  def reward_spec(self) -> specs.DiscreteArraySpec:
    """Describes the reward returned by the env."""
    return specs.DiscreteArraySpec(num_values=2)

  def observation_spec(self) -> specs.BoundedArraySpec:
    """Defines the observations provided by the env."""
    return specs.BoundedArraySpec(shape=(3,), dtype=int, minimum=0, maximum=1)

  def action_spec(self) -> specs.DiscreteArraySpec:
    """Defines the actions that should be provided to `step`."""
    return specs.DiscreteArraySpec(num_values=2)


class DiscountingEnv(worlds.Environment):
  """Used to test a learner's ability to perform value discounting.

  This environment has an episode length of three, ending in a reward. The agent
  must be able to correctly assign the discounted value across the preceding states:
      V([1, 0, 0]) = gamma            V([0, 1, 0]) = 1.
  """

  def __init__(self) -> None:
    """Initializer."""
    super().__init__()
    self._t = 0

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    self._t = 0
    return worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=[1, 0, 0])

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    del action
    timestep = None
    if self._t == 0:
      timestep = worlds.TimeStep(step_type=worlds.StepType.MID, reward=0, observation=[0, 1, 0])
    elif self._t == 1:
      timestep = worlds.TimeStep(step_type=worlds.StepType.LAST, reward=1, observation=[0, 0, 1])
    else:
      raise RuntimeError("Internal time index exceeded.")
    self._t += 1
    return timestep

  def reward_spec(self) -> specs.DiscreteArraySpec:
    """Describes the reward returned by the env."""
    return specs.DiscreteArraySpec(num_values=2)

  def observation_spec(self) -> specs.BoundedArraySpec:
    """Defines the observations provided by the env."""
    return specs.BoundedArraySpec(shape=(3,), dtype=int, minimum=0, maximum=1)

  def action_spec(self) -> specs.DiscreteArraySpec:
    """Defines the actions that should be provided to `step`."""
    return specs.DiscreteArraySpec(num_values=1)
