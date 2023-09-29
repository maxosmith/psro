"""Strategy objects defining containers of policies and a distribution over them."""
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from marl import individuals, types, worlds

Mixture = Sequence[float]
PureProfile = Mapping[types.PlayerID, types.PolicyID]
MixedProfile = Mapping[types.PlayerID, Mixture]
Profile = Union[PureProfile, MixedProfile]
ProfilePayoffs = Mapping[types.PlayerID, types.Array]
ProfilePayoffSummary = Mapping[types.PlayerID, float]


def is_probability_distribution(p: types.Array) -> bool:
  """Checks if an array is a valid probability distribution."""
  p = np.asarray(p)
  return np.all(0 <= p) and np.all(p <= 1) and np.isclose(p.sum(), 1)


class Strategy(individuals.Bot):
  """A collection of player policies and a distribution over their episodic play.

  Args:
    policies: Policies contained in the strategy set. Their position in this sequence
      is referred to as their `PolicyID`.
    mixture: Distribution over policies. Indices in `mixture` must correspond to
      the policy at the same index in `policies`.
    seed: Random number generator seed.
  """

  def __init__(
      self,
      policies: Sequence[individuals.Individual],
      mixture: Sequence[float],
      seed: int | None = None,
  ):
    """Initializer."""
    self._policies = policies
    self._rng = np.random.default_rng(seed)
    self._policy = None
    self.mixture = mixture

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.State, types.Action]:
    """Select an action to take given the current timestep."""
    assert self._policy, "Episode-reset must be called before step."
    return self._policy.step(state=state, timestep=timestep)

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Reset the state of the agent at the start of an epsiode."""
    policy_id = self._rng.choice(len(self._policies), p=self._mixture)
    self._policy = self._policies[policy_id]
    return self._policy.episode_reset(timestep=timestep)

  @property
  def mixture(self) -> Mixture:
    """Mixture getter."""
    return self._mixture

  @mixture.setter
  def mixture(self, new_mixture: Mixture):
    """Mixture setter."""
    if not is_probability_distribution(new_mixture):
      raise ValueError("Mixture must be a valid probability distribution.")
    self._mixture = new_mixture
