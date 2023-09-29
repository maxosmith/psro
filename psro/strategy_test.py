"""Test for `psro.strategy`."""
import dataclasses
from typing import Tuple

from absl.testing import absltest, parameterized

from marl import types, worlds
from psro import strategy


@dataclasses.dataclass
class FakePolicy:
  """Fake policy for testing."""

  action: int

  def step(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.State, types.Action]:
    """Select an action to take given the current timestep."""
    del timestep
    return state, self.action

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Reset the state of the agent at the start of an epsiode."""
    del timestep
    return ()


class StrategyTest(parameterized.TestCase):
  """Test suite for `psro.strategy.Strategy`."""

  def test_mixture(self):
    """Tests `set_policy`."""
    player = strategy.Strategy(
        policies=[FakePolicy(x) for x in range(3)],
        mixture=[1, 0, 0],
    )

    for policy_i in range(3):
      mixture = [0, 0, 0]
      mixture[policy_i] = 1
      player.mixture = mixture
      player.episode_reset(None)
      _, action = player.step(None, None)
      self.assertEqual(action, policy_i)

  @parameterized.parameters(
      {"mixture": [0.8, 0.1]},
      {"mixture": [0.00001, 0.99]},
  )
  def test_invalid_mixture(self, mixture: strategy.Mixture):
    """Tests that mixtures must be valid probability distributions."""
    with self.assertRaises(ValueError):
      strategy.Strategy(
          policies=[FakePolicy(None)] * len(mixture),
          mixture=mixture,
      )


if __name__ == "__main__":
  absltest.main()
