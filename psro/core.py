"""Core PSRO definitions."""
import pathlib
from typing import Callable, Mapping, NamedTuple

from marl import individuals, types, worlds

from psro import strategy

Epoch = int
LearnerID = types.PlayerID
Solution = Mapping[types.PlayerID, types.Array]
GameCtor = Callable[[], worlds.Game]
GameMatrix = types.Array
GameSolver = Callable[[GameMatrix], Solution]


class ResponseOracleJob(NamedTuple):
  """Response oracle job description."""

  learner_id: LearnerID
  players: strategy.JointStrategy
  game_ctor: Callable[[], worlds.Game]
  epoch_dir: str | pathlib.Path


ResponseOracle = Callable[[ResponseOracleJob], individuals.Bot]


class SimulationJob(NamedTuple):
  """Profile simulation strategy job description."""

  game_ctor: worlds.Game
  players: strategy.JointStrategy
  profile: strategy.PureProfile
  num_episodes: int


ProfileSimulator = Callable[[SimulationJob], tuple[strategy.Profile, types.Tree]]
