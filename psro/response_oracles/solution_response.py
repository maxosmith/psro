"""Computes a response to the current solution."""
import os

from psro import core


def oracle(job: core.ResponseOracleJob):
  """Computes a new policy via an oracle method."""
  player_dir = job.epoch_dir / f"player_{job.learner_id}"
  os.mkdir(player_dir)
  coplayers = {id: strategy for id, strategy in job.players.items() if id != job.learner_id}
  return (job.learner_id, job.oracle(job.learner_id, job.game_ctor, coplayers, player_dir))
