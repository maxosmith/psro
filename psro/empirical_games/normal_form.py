"""Empirical normal-form game."""
import dataclasses
import itertools
import pathlib
import shelve
from typing import Callable, Sequence

import numpy as np
from marl import types

from psro import strategy

AggregateFn = Callable[[Sequence[float]], float]


@dataclasses.dataclass
class NormalForm:
  """A empirical normal-form game representing payoffs for different strategy profiles.

  Args:
    num_players: Number of players in the game.
    path: Optional filepath to disk-back the game data.
  """

  num_players: int
  path: str | pathlib.Path | None = None

  def __post_init__(self):
    """Initialize payoffs and policy counters."""
    self.payoffs = shelve.open(str(self.path)) if self.path else {}
    self.num_policies = {}

  def __del__(self):
    """Ensure payoffs are closed if file-backed."""
    if self.path:
      self.payoffs.close()

  def add_payoffs(self, profile: strategy.PureProfile, payoffs: strategy.ProfilePayoffs):
    """Add payoff for a given strategy profile.

    Args:
      profile: Profile associated with the payoffs.
      payoffs: Payoffs per-player to record.
    """
    key = self._profile_to_key(profile)

    # Update the number of policies for each agent
    self.num_policies = {agent: max(self.num_policies.get(agent, 0), policy + 1) for agent, policy in profile.items()}

    # Update payoffs for the given profile
    if key in self.payoffs:
      for agent in self.payoffs[key].keys():
        self.payoffs[key][agent] = np.append(self.payoffs[key][agent], payoffs[agent])
    else:
      self.payoffs[key] = payoffs

  def average_payoffs(
      self,
      profile: strategy.Profile,
      tolerance: float | None = 10**-7,
  ) -> strategy.ProfilePayoffSummary:
    """Compute the expected payoffs for a strategy profile.

    Args:
      profile: Strategy profiles to look up their average payoffs.
      tolerance: Minimum tolerance of probability for a specific profile. Mixed strategy profiles
        can contain many pure strategy profiles of exceptionally low support.

    Returns:
      Average payoffs for the strategy profile.
    """
    if all(isinstance(p, types.PolicyID) for p in profile.values()):
      return {agent: np.mean(payoff) for agent, payoff in self.payoff_samples(profile).items()}

    # Convert pure-strategy profiles to a mixture for consistency in later calculations
    profile = {
        agent: np.array([1.0 if i == agent_profile else 0.0 for i in range(self.num_policies[agent])])
        if isinstance(agent_profile, types.PolicyID)
        else agent_profile
        for agent, agent_profile in profile.items()
    }

    # Calculate average payoff for mixed strategies
    total_payoff = {agent_id: 0.0 for agent_id in range(self.num_players)}
    for possible_profile in itertools.product(*[np.arange(len(profile[i])) for i in range(self.num_players)]):
      possible_profile_dict = dict(enumerate(possible_profile))
      coeff = np.prod([profile[agent_id][policy_id] for agent_id, policy_id in possible_profile_dict.items()])

      if tolerance and (coeff < tolerance):
        continue

      payoffs = self.average_payoffs(possible_profile_dict)
      total_payoff = {agent: total_payoff[agent] + coeff * payoffs[agent] for agent in payoffs}

    return total_payoff

  def payoff_samples(self, profile: strategy.PureProfile) -> strategy.ProfilePayoffs:
    """Retrieve the simulated payoffs for a given strategy profile.

    Args:
      profile: Pure strategy profile to lookup payoff samples.

    Returns:
      All payoffs recorded for the profile.
    """
    key = self._profile_to_key(profile)
    return self.payoffs.get(key, None)

  def num_samples(self, profile: strategy.PureProfile) -> int:
    """Count the number of samples for a given profile.

    Args:
      profile: Pure strategy profile to query.

    Returns:
      Number of payoff samples recorded for this profile.
    """
    key = self._profile_to_key(profile)
    return min(len(x) for x in self.payoffs[key].values()) if key in self.payoffs else 0

  def game_matrix(self, aggregate_fn: AggregateFn = np.mean) -> np.ndarray:
    """Get the matrix-form representation.

    NOTE: This currently requires that the empirical game is complete.

    Returns:
      Game matrix containing average payoffs.
    """
    num_policies = [self.num_policies[i] for i in range(self.num_players)]
    matrix = np.zeros(num_policies + [self.num_players], dtype=np.float32)
    for profile in itertools.product(*[np.arange(x) for x in num_policies]):
      key = self._profile_to_key(dict(enumerate(profile)))
      payoffs = self.payoffs[key]
      payoffs = [aggregate_fn(payoffs[i]) for i in range(self.num_players)]
      matrix[tuple(profile)] = payoffs
    return matrix

  def _profile_to_key(self, profile: strategy.PureProfile) -> str:
    """Get the dict key for the strategy profile."""
    key = ""
    for agent in range(self.num_players):
      key += f"x{profile[agent]}"
    return key

  def __contains__(self, profile: strategy.PureProfile) -> bool:
    """Check if this has payoffs for a profile.

    Args:
        profile: profile to check

    Returns:
        true if there is at least one sample of the payoff.
    """
    key = self._profile_to_key(profile)
    return key in self.payoffs
