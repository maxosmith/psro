"""Utilities for testing."""
import dataclasses
from typing import Sequence

import numpy as np
from marl import individuals, specs, types, worlds

from psro import core, strategy


@dataclasses.dataclass
class StubProfileSimulator:
  """Returns simulation profiles matching a predefined game matrix."""

  payoff_matrix: types.Array

  def __call__(self, job: core.SimulationJob) -> tuple[strategy.Profile, types.Tree]:
    """Simulate a single profile."""
    payoff_indices = [job.profile[player] for player in range(len(job.profile))]
    payoffs = {i: np.full([job.num_episodes], v) for i, v in enumerate(self.payoff_matrix[tuple(payoff_indices)])}
    return (job.profile, payoffs)


@dataclasses.dataclass
class MockGameSolver:
  """Mock game solver."""

  expected_games: Sequence[core.GameMatrix]
  solutions: Sequence[core.Solution]

  def __post_init__(self):
    """Post initializer."""
    self._t = 0

  def __call__(self, game_matrix: core.GameMatrix) -> core.Solution:
    """Solve the game."""
    np.testing.assert_equal(game_matrix, self.expected_games[self._t])
    solution = self.solutions[self._t]
    self._t += 1
    return solution


@dataclasses.dataclass
class StubResponseOracle:
  """Mock response oracle.

  ResponseOracles cannot be trivially mocked, because they're evoked using a
  ProcessPoolExecutor and their state is consistent across calls per epoch.
  """

  outputs: Sequence[individuals.Bot]

  def __post_init__(self):
    """Post initializer."""
    self._epoch = 0

  def __call__(self, job: core.ResponseOracleJob) -> individuals.Bot:
    """Compute new response policy."""
    self._epoch += 1
    return (job.learner_id, self.outputs[self._epoch - 1])


class DummyGame(worlds.Game):
  """Dummy Game for testing."""

  def reset(self) -> worlds.PlayerIDToTimestep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    raise NotImplementedError()

  def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    del actions
    raise NotImplementedError()

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the game to each player."""
    raise NotImplementedError()

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Defines the observations provided by the game to each player."""
    raise NotImplementedError()

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Defines the actions that should be provided to `step` by each player."""
    raise NotImplementedError()
