"""."""
import contextlib
import os
import pathlib
from typing import Any
from unittest.mock import patch

import cloudpickle
from marl import individuals

from psro import core
from psro.response_oracles.cleanrl_proxy import handlers

Patch = tuple[str, Any]


@contextlib.contextmanager
def _change_dir(destination):
  """Temporarily changes the current working directory."""
  current_dir = os.getcwd()  # Save the current working directory
  os.chdir(destination)  # Change to the new directory
  try:
    yield
  finally:
    os.chdir(current_dir)  # Change back to the original directory


class CleanRLProxy:
  """Compute approximate best-responses through CleanRL.

  Args:
    handler: Handler for the specific CleanRL algorithm that will be run.
  """

  def __init__(self, handler: handlers.Handler):
    """Initializer."""
    self._handler = handler

  def __call__(self, job: core.ResponseOracleJob) -> individuals.Bot:
    """Run an approximate best-response calculation."""
    # Build patches that allow us to modify CleanRL.
    patch_specs = self._handler.build_patches(job)

    # Build a result directory. We'll also change into this directory for the duration
    # of the best-response calculation in order to capture all of the CleanRL artifacts.
    player_dir = pathlib.Path(job.epoch_dir) / f"player_{job.learner_id}"
    os.makedirs(player_dir, exist_ok=True)

    # Compute the approximate best response.
    with contextlib.ExitStack() as stack, _change_dir(player_dir):
      patches = [stack.enter_context(patch(target, **kwargs)) for target, kwargs in patch_specs]
      self._handler.run()
      run_dir = self._get_cleanrl_run_dir()
    del patches

    # Load and build the approximate best-response from the CleanRL artifacts.
    bot = self._handler.build_bot(job, run_dir)
    bot_path = player_dir / "policy.pb"
    with open(bot_path, "wb") as file:
      cloudpickle.dump(bot, file)
    return (job.learner_id, bot)

  def _get_cleanrl_run_dir(self) -> pathlib.Path:
    """Get the directory that CleanRL wrote results into.

    NOTE: This assumes that you've called this function from the same context that
      was used to call CleanRL.
    """
    run_dirs = [name for name in os.listdir("runs/") if os.path.isdir(os.path.join("runs/", name))]
    assert len(run_dirs) == 1, f"Found {len(run_dirs)} runs, when exactly one expected."
    return pathlib.Path(os.path.abspath(os.path.join("runs", run_dirs[0])))
