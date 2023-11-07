"""Core PSRO definitions."""
import pathlib
from typing import Any, Callable, Mapping, NamedTuple

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
  solution: strategy.MixedProfile
  epoch_dir: str | pathlib.Path
  extras: Any | None = None


ResponseOracle = Callable[[ResponseOracleJob], individuals.Bot]


class SimulationJob(NamedTuple):
  """Profile simulation strategy job description."""

  game_ctor: worlds.Game
  players: strategy.JointStrategy
  profile: strategy.PureProfile
  num_episodes: int
  extras: Any | None = None


ProfileSimulator = Callable[[SimulationJob], tuple[strategy.Profile, types.Tree]]
