"""Test for `test_envs`."""
import copy

from absl.testing import absltest, parameterized
from marl import worlds

from psro.utils import test_envs


class TestEnvsTest(parameterized.TestCase):
  """Test suite for `test_envs`."""

  def test_reward_env(self):
    """Regression test of `RewardEnv."""
    env = test_envs.RewardEnv()
    self.assertEqual(env.observation_spec().num_values, 2)
    self.assertEqual(env.reward_spec().num_values, 2)
    self.assertEqual(env.action_spec().num_values, 1)

    ts = env.reset()
    self.assertEqual(ts.observation, 0)
    self.assertEqual(ts.reward, 1)
    self.assertEqual(ts.step_type, worlds.StepType.FIRST)

    ts = env.step(0)
    self.assertEqual(ts.observation, 1)
    self.assertEqual(ts.reward, 0)
    self.assertEqual(ts.step_type, worlds.StepType.LAST)

  @parameterized.parameters((1, 1, 1), (0, 2, 0))
  def test_action_value_env(self, action: int, observation: int, reward: int):
    """Regression test of `ActionValueEnv."""
    env = test_envs.ActionValueEnv()
    self.assertEqual(env.observation_spec().num_values, 3)
    self.assertEqual(env.reward_spec().num_values, 2)
    self.assertEqual(env.action_spec().num_values, 2)

    ts = env.reset()
    self.assertEqual(ts.observation, 0)
    self.assertEqual(ts.reward, 0)
    self.assertEqual(ts.step_type, worlds.StepType.FIRST)

    ts = env.step(action)
    self.assertEqual(ts.observation, observation)
    self.assertEqual(ts.reward, reward)
    self.assertEqual(ts.step_type, worlds.StepType.LAST)

  def test_state_value_env(self):
    """Regression test of `StateValueEnv`."""
    env = test_envs.StateValueEnv()
    self.assertEqual(env.observation_spec().num_values, 3)
    self.assertEqual(env.reward_spec().num_values, 2)
    self.assertEqual(env.action_spec().num_values, 1)

    for _ in range(50):  # Run several episodes to verify stochastic outcomes.
      ts = env.reset()
      observation = ts.observation
      if observation == 0:
        self.assertEqual(ts.reward, 0)
      elif observation == 1:
        self.assertEqual(ts.reward, 1)
      else:
        raise RuntimeError(f"Unexpected observation {observation}.")
      self.assertEqual(ts.step_type, worlds.StepType.FIRST)

      ts = env.step(0)
      self.assertEqual(ts.observation, 2)
      self.assertEqual(ts.reward, 0)
      self.assertEqual(ts.step_type, worlds.StepType.LAST)

  def test_state_action_value_env(self):
    """Regression test of `StateActionValueEnv`."""
    env = test_envs.StateActionValueEnv()
    self.assertEqual(env.observation_spec().num_values, 3)
    self.assertEqual(env.reward_spec().num_values, 2)
    self.assertEqual(env.action_spec().num_values, 2)

    for _ in range(50):  # Run several episodes to verify stochastic outcomes.
      ts = env.reset()
      observation = ts.observation
      if observation not in [0, 1]:
        raise RuntimeError(f"Unexpected observation {observation}.")
      self.assertEqual(ts.reward, 0)
      self.assertEqual(ts.step_type, worlds.StepType.FIRST)

      # Action 0 is only reward when the observation is 0.
      env0 = copy.deepcopy(env)
      ts = env0.step(0)
      if observation:
        self.assertEqual(ts.observation, 2)
        self.assertEqual(ts.reward, 0)
      else:
        self.assertEqual(ts.observation, 2)
        self.assertEqual(ts.reward, 1)
      self.assertEqual(ts.step_type, worlds.StepType.LAST)

      # Action 1 is only reward when the observation is 1.
      env1 = copy.deepcopy(env)
      ts = env1.step(1)
      if observation:
        self.assertEqual(ts.observation, 2)
        self.assertEqual(ts.reward, 1)
      else:
        self.assertEqual(ts.observation, 2)
        self.assertEqual(ts.reward, 0)
      self.assertEqual(ts.step_type, worlds.StepType.LAST)

  def test_discounting_env(self):
    """Regression test of `DiscountingEnv`."""
    env = test_envs.DiscountingEnv()
    self.assertEqual(env.observation_spec().num_values, 3)
    self.assertEqual(env.reward_spec().num_values, 2)
    self.assertEqual(env.action_spec().num_values, 1)

    ts = env.reset()
    self.assertEqual(ts.observation, 0)
    self.assertEqual(ts.reward, 0)
    self.assertEqual(ts.step_type, worlds.StepType.FIRST)

    ts = env.step(0)
    self.assertEqual(ts.observation, 1)
    self.assertEqual(ts.reward, 0)
    self.assertEqual(ts.step_type, worlds.StepType.MID)

    ts = env.step(0)
    self.assertEqual(ts.observation, 2)
    self.assertEqual(ts.reward, 1)
    self.assertEqual(ts.step_type, worlds.StepType.LAST)


if __name__ == "__main__":
  absltest.main()
