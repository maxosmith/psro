from typing import Sequence

import numpy as np

from psro import strategy


class Uniform:
  """Computes a uniform distribution over policies."""

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Compute PRD for the given payoff matrix.

    Args:
      payoffs: Payoff matrix [P1, P2, ...., PN, NumPlayers].

    Returns:
      Computed solution.
    """
    del kwargs
    strategies = (np.ones(x, dtype=float) / x for x in payoffs.shape[:-1])
    return dict(enumerate(strategies))


class UniformBiased:
  """Computes a biased uniform distribution over policies.

  The uniform distribution is biased to prioritize playing against more recent
  policies (Policies that were appended to the policy list later in training)
  instead of older ones.
  """

  @staticmethod
  def softmax_on_range(num_policies: int) -> np.ndarray:
    """Compute the biased uniform strategy."""
    x = np.array(list(range(num_policies)))
    x = np.exp(x - x.max())
    x /= np.sum(x)
    return x

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Compute PRD for the given payoff matrix.

    Args:
      payoffs: Payoff matrix [P1, P2, ...., PN, NumPlayers].

    Returns:
      Computed solution.
    """
    del kwargs
    strategies = (UniformBiased.softmax_on_range(x) for x in payoffs.shape[:-1])
    return dict(enumerate(strategies))
