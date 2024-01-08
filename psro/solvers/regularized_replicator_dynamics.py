"""Regularized Replicator Dynamics."""
import collections
import dataclasses
from typing import Sequence

import numpy as np
import sox
import tree
from open_spiel.python.algorithms import projected_replicator_dynamics as os_prd

from psro import strategy


@dataclasses.dataclass
class RegularizedReplicatorDynamics:
  """Regularized Replicator Dynamics (RRD).

  Args:
    min_iterations: minimum number of replicator iterations to perform.
    max_iterations: maximum number of replicator iterations to perform.
    dt: Update amplitude term.
    gamma: Minimum exploratory probability term.
    average_strategies: whether to return the final strategy (False), or
      return an average over the last `average_strategies_window_length` strategies.
    average_strategies_window_length: Number of strategies to average.
    use_approx: use the approximate simplex projection.
  """

  min_iterations: int = 10_000
  max_iterations: int = 100_000
  regret_threshold: float = 0.35
  dt: float = 1e-3
  gamma: float = 1e-6
  average_strategies: bool = False
  average_strategies_window_length: None | int = None
  use_approx: bool = False

  def __post_init__(self):
    """Post initializer."""
    if (self.average_strategies_window_length is not None) and (not self.average_strategies):
      raise ValueError("Window length set for strategy averaging while it's disabled.")

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Compute PRD for the given payoff matrix.

    Args:
      payoffs: Payoff matrix [P1, P2, ...., PN, NumPlayers].

    Returns:
      Computed solution.
    """
    payoff_tensors = sox.array_utils.unstack(payoffs, axis=-1)
    number_players = len(payoff_tensors)
    action_space_shapes = payoff_tensors[0].shape  # Number of actions available to each player.

    new_strategies = [np.ones(action_space_shapes[k]) / action_space_shapes[k] for k in range(number_players)]

    average_strategy_window = collections.deque(maxlen=self.average_strategies_window_length or self.max_iterations)
    for i in range(self.max_iterations):
      new_strategies = os_prd._projected_replicator_dynamics_step(
          payoff_tensors, new_strategies, self.dt, self.gamma, self.use_approx
      )
      average_strategy_window.append(new_strategies)

      if (i > self.min_iterations) and (np.sum(compute_regret(payoffs, new_strategies)) <= self.regret_threshold):
        break

    if self.average_strategies:
      average_new_strategies = tree.map_structure(lambda *x: np.mean(x, axis=0), *average_strategy_window)
      return [dict(enumerate(average_new_strategies))]
    else:
      return [dict(enumerate(new_strategies))]


def compute_regret(payoff_tensor: np.ndarray, player_strategies: list[np.ndarray]) -> np.ndarray:
  """Compute per-player regret.

  Args:
    payoff_tensor: Payoff matrix [P1, P2, ...., PN, NumPlayers].
    player_strategies: List of player strategies (distributions over policies) [[P1], [P2], ..., [PN]].

  Returns:
    Per-player regrets [NumPlayers].
  """
  num_players = len(player_strategies)

  # Validate dimensions
  if payoff_tensor.shape[:-1] != tuple(s.shape[0] for s in player_strategies):
    raise ValueError("Dimensions of payoff tensor and player strategies do not match.")

  # Compute actual payoff
  actual_payoff = payoff_tensor.copy()
  for i, strat in enumerate(player_strategies):
    actual_payoff = np.tensordot(actual_payoff, strat, axes=([0], [0]))

  # Initialize array to store regrets
  regrets = np.zeros(num_players)

  # Compute best possible payoff and regret for each player
  for i, _ in enumerate(player_strategies):
    # Fix strategies for all players except player i
    best_response_payoff = np.moveaxis(payoff_tensor[..., i], i, -1)

    for j, other_strategy in enumerate(player_strategies):
      if i == j:
        continue
      best_response_payoff = np.tensordot(best_response_payoff, other_strategy, axes=([0], [0]))
    # Find the best possible payoff for player i
    best_payoff = np.max(best_response_payoff)
    # Compute regret
    regrets[i] = best_payoff - actual_payoff[i]

  return regrets
