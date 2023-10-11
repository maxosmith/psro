"""Simulate MARL Bots."""
from typing import List

import numpy as np
import tree
from marl import types
from marl.services.arenas import sim_arena

from psro import core, strategy


def simulate_profile(job: core.SimulationJob) -> tuple[strategy.Profile, types.Tree]:
  """Simulates a strategy profile for a number of episodes."""
  players = {}
  for player_id, player_strategy in job.players.items():
    mixture = np.zeros_like(player_strategy.mixture)
    mixture[job.profile[player_id]] = 1.0
    player_strategy.mixture = mixture
    players[player_id] = player_strategy

  arena = sim_arena.SimArena(job.game_ctor())
  episodic_results: List[sim_arena.EpisodeResult] = arena.run(players=players, num_episodes=job.num_episodes)
  episodic_results: sim_arena.EpisodeResult = tree.map_structure(lambda *args: np.stack(args), *episodic_results)
  return (job.profile, episodic_results.episode_return)
