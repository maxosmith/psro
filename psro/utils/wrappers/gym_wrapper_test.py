"""Test suite for gymnasium wrappers and functions."""
import chex
import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from marl import specs, worlds

from psro.utils.wrappers import gym_wrapper


class FromGymnasiumTest(parameterized.TestCase):
  """Test suite for `FromGymnasium`."""

  def test_cartpole(self):
    """Test converting cartpole."""
    env = gym.make("CartPole-v1", max_episode_steps=3)
    converted = gym_wrapper.FromGymnasium(env)

    obs_spec = converted.observation_spec()
    np.testing.assert_almost_equal(env.observation_space.low, obs_spec.minimum)
    np.testing.assert_almost_equal(env.observation_space.high, obs_spec.maximum)

    action_spec = converted.action_spec()
    np.testing.assert_almost_equal(env.action_space.n, action_spec.num_values)

    o, _ = env.reset()
    timestep = converted.reset()
    chex.assert_equal_shape([o, timestep.observation])
    self.assertEqual(worlds.StepType.FIRST, timestep.step_type)

    o, r, *_ = env.step(0)
    timestep = converted.step(0)
    chex.assert_equal_shape([o, timestep.observation])
    chex.assert_equal_shape([np.asarray(r), np.asarray(timestep.reward)])
    self.assertEqual(worlds.StepType.MID, timestep.step_type)

    o, r, *_ = env.step(0)
    timestep = converted.step(0)
    chex.assert_equal_shape([o, timestep.observation])
    chex.assert_equal_shape([np.asarray(r), np.asarray(timestep.reward)])
    self.assertEqual(worlds.StepType.LAST, timestep.step_type)


class ToGymnasiumTest(parameterized.TestCase):
  """Test suite for `ToGymnasium`."""

  def test_cartpole(self):
    """Test converting cartpole."""
    env = gym_wrapper.FromGymnasium(gym.make("CartPole-v1", max_episode_steps=3))
    converted = gym_wrapper.ToGymnasium(env)

    np.testing.assert_almost_equal(env.observation_spec().minimum, converted.observation_space.low)
    np.testing.assert_almost_equal(env.observation_spec().maximum, converted.observation_space.high)

    np.testing.assert_almost_equal(converted.action_space.n, env.action_spec().num_values)

    timestep = env.reset()
    o, _ = converted.reset()
    chex.assert_equal_shape([o, timestep.observation])
    self.assertEqual(worlds.StepType.FIRST, timestep.step_type)

    timestep = env.step(0)
    o, r, term, trunc, _ = converted.step(0)
    chex.assert_equal_shape([o, timestep.observation])
    chex.assert_equal_shape([np.asarray(r), np.asarray(timestep.reward)])
    self.assertTrue(isinstance(term, bool))
    self.assertTrue(isinstance(trunc, bool))


class SpaceConversionTest(parameterized.TestCase):
  """Test suite for `spec_to_space` and `space_to_spec` functions."""

  @parameterized.parameters(
      (specs.ArraySpec(shape=(3,), dtype=np.int32), gym.spaces.Box),
      (specs.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0), gym.spaces.Box),
      (specs.DiscreteArraySpec(num_values=4, dtype=int), gym.spaces.Discrete),
  )
  def test_spec_to_space(self, spec, expected_type):
    """Tests conversion from bsuite spec to gymnasium space."""
    gym_space = gym_wrapper.spec_to_space(spec)
    self.assertIsInstance(gym_space, expected_type)

  @parameterized.parameters(
      (gym.spaces.Box(low=-1.0, high=1.0, shape=(2,)), specs.BoundedArraySpec),
      (gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(3,)), specs.ArraySpec),
      (gym.spaces.Discrete(4), specs.DiscreteArraySpec),
  )
  def test_space_to_spec(self, space, expected_type):
    """Tests conversion from gymnasium space to bsuite spec."""
    bsuite_spec = gym_wrapper.space_to_spec(space)
    self.assertIsInstance(bsuite_spec, expected_type)


if __name__ == "__main__":
  absltest.main()
