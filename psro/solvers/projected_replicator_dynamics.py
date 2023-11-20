"""Proxy for OpenSpiel's Projected Replicator Dynamics."""
import dataclasses
from typing import Sequence

import numpy as np
import sox
import tree
from open_spiel.python.algorithms import projected_replicator_dynamics as os_prd

from psro import strategy


@dataclasses.dataclass
class ProjectedReplicatorDynamics:
  """Projected Replicator Dynamics (PRD).

  Args:
  """

  num_iterations: int = 10_000
  dt: float = 1e-3
  gamma: float = 1e-6
  average_over_last_n_strategies: None | int = None
  use_approx: bool = False

  def __call__(self, payoffs: np.ndarray, **kwargs) -> Sequence[strategy.Profile]:
    """."""
    payoff_tensors = sox.array_utils.unstack(payoffs, axis=-1)

    number_players = len(payoff_tensors)
    # Number of actions available to each player.
    action_space_shapes = payoff_tensors[0].shape

    # If no initial starting position is given, start with uniform probabilities.
    # TODO(max): Removed the ability to set optional initial strategies.
    new_strategies = [np.ones(action_space_shapes[k]) / action_space_shapes[k] for k in range(number_players)]

    average_over_last_n_strategies = self.average_over_last_n_strategies or self.num_iterations

    meta_strategy_window = []
    for i in range(self.num_iterations):
      new_strategies = os_prd._projected_replicator_dynamics_step(
          payoff_tensors, new_strategies, self.dt, self.gamma, self.use_approx
      )
      if i >= self.num_iterations - average_over_last_n_strategies:
        meta_strategy_window.append(new_strategies)
    average_new_strategies = tree.map_structure(lambda *x: np.mean(x, axis=0), *meta_strategy_window)
    return dict(enumerate(average_new_strategies))
