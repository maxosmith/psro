import collections
import dataclasses
from typing import Sequence

import numpy as np
import sox
from open_spiel.python.algorithms import regret_matching as os_rm

from psro import strategy

# Start with initial regrets of 1 / denom.
INITIAL_REGRET_DENOM = 1e6


@dataclasses.dataclass
class RegretMatching:
  """Proxy to OpenSpiel's regret matching algorithm."""

  iterations: int = 100_000
  gamma: float = 1e-6
  average_over_last_n_strategies: None | int = None

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """Compute PRD for the given payoff matrix.

    Args:
      payoffs: Payoff matrix [P1, P2, ...., PN, NumPlayers].

    Returns:
      Computed solution.
    """
    payoff_tensors = sox.array_utils.unstack(payoffs, axis=-1)
    number_players = len(payoff_tensors)
    # Number of actions available to each player.
    action_space_shapes = payoff_tensors[0].shape

    # If no initial starting position is given, start with uniform probabilities.
    new_strategies = [np.ones(action_space_shapes[k]) / action_space_shapes[k] for k in range(number_players)]

    regrets = [np.ones(action_space_shapes[k]) / INITIAL_REGRET_DENOM for k in range(number_players)]

    averager = StrategyAverager(number_players, action_space_shapes, self.average_over_last_n_strategies)
    averager.append(new_strategies)

    for _ in range(self.iterations):
      new_strategies = os_rm._regret_matching_step(payoff_tensors, new_strategies, regrets, self.gamma)
      averager.append(new_strategies)

    result = averager.average_strategies()

    return [dict(enumerate(result))]


class StrategyAverager(object):
  """A helper class for averaging strategies for players."""

  def __init__(self, num_players, action_space_shapes, window_size=None):
    """Initialize the average strategy helper object.

    Args:
      num_players (int): the number of players in the game,
      action_space_shapes:  an vector of n integers, where each element
          represents the size of player i's actions space,
      window_size (int or None): if None, computes the players' average
          strategies over the entire sequence, otherwise computes the average
          strategy over a finite-sized window of the k last entries.
    """
    self._num_players = num_players
    self._action_space_shapes = action_space_shapes
    self._window_size = window_size
    self._num = 0
    if self._window_size is None:
      self._sum_meta_strategies = [np.zeros(action_space_shapes[p]) for p in range(num_players)]
    else:
      self._window = collections.deque(maxlen=self._window_size)

  def append(self, meta_strategies):
    """Append the meta-strategies to the averaged sequence.

    Args:
      meta_strategies: a list of strategies, one per player.
    """
    if self._window_size is None:
      for p in range(self._num_players):
        self._sum_meta_strategies[p] += meta_strategies[p]
    else:
      self._window.append(meta_strategies)
    self._num += 1

  def average_strategies(self):
    """Return each player's average strategy.

    Returns:
      The averaged strategies, as a list containing one strategy per player.
    """

    if self._window_size is None:
      avg_meta_strategies = [np.copy(x) for x in self._sum_meta_strategies]
      num_strategies = self._num
    else:
      avg_meta_strategies = [np.zeros(self._action_space_shapes[p]) for p in range(self._num_players)]
      for i in range(len(self._window)):
        for p in range(self._num_players):
          avg_meta_strategies[p] += self._window[i][p]
      num_strategies = len(self._window)
    for p in range(self._num_players):
      avg_meta_strategies[p] /= num_strategies
    return avg_meta_strategies
