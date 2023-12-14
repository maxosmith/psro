"""Algorithm handler interface."""
import abc
import importlib
import pathlib
from typing import Any

from marl import individuals

from psro import core

Patch = tuple[str, Any]


class Handler(metaclass=abc.ABCMeta):
  """Handles building patches and the resultant bot for specific algorithms.

  Attrs:
    name: Name corresponding to the algorithm's file in CleanRL.
  """

  name: str

  def __init__(self) -> None:
    """Initializer."""

  @abc.abstractmethod
  def build_patches(self, job: core.ResponseOracleJob) -> list[Patch]:
    """Build patches that enable the main program to be run as a subroutine."""

  @abc.abstractmethod
  def build_bot(self, job: core.ResponseOracleJob, run_dir: pathlib.Path) -> individuals.Bot:
    """Build a bot from a completed job."""

  def run(self):
    """Run the CleanRL algorithm."""
    importlib.import_module(f"cleanrl.{self.name}").main()
