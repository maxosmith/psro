"""Utilities for interacting with PSRO results."""
import itertools
import os
import os.path as osp
import pathlib

import cloudpickle
import numpy as np

from psro import strategy


def load_num_epochs(path: pathlib.Path):
  """loads the number of valid epochs saved in a result directory.

  A valid epoch directory must contain both 'game_matrix.pb' and 'solution.pb'.

  Args:
    path: The path to the directory containing epoch folders.

  Returns:
    The largest epoch number found in the directory among valid epoch directories.
  """
  # Ensure the directory path is a string, in case a Path object is passed
  path = str(path)

  # Get all the items in the directory
  items = os.listdir(path)

  # Initialize the largest epoch number to 0
  largest_epoch_number = 0

  # Iterate over all items and find the largest epoch number
  for item in items:
    # Check if the item starts with 'epoch_' and is a directory
    if item.startswith("epoch_") and os.path.isdir(os.path.join(path, item)):
      epoch_dir = os.path.join(path, item)
      # Check if both 'game_matrix.pb' and 'solution.pb' exist in the epoch directory
      if os.path.isfile(os.path.join(epoch_dir, "game_matrix.pb")) and os.path.isfile(
          os.path.join(epoch_dir, "solution.pb")
      ):
        try:
          # Extract the epoch number and update the largest epoch number if necessary
          epoch_number = int(item.split("_")[1])
          largest_epoch_number = max(largest_epoch_number, epoch_number)
        except ValueError:
          # If the conversion to integer fails, ignore the item
          pass

  return largest_epoch_number


def load_solutions(path: pathlib.Path) -> list[strategy.Profile]:
  """Load the solutions computed in a run of PSRO.

  Args:
    path: Result directory from a run of PSRO.

  Returns:
    List of the solutions computed in the run.
  """
  path = pathlib.Path(path)
  solutions = []
  for epoch in itertools.count(1):
    epoch_dir = path / f"epoch_{epoch}"
    solution_path = epoch_dir / "solution.pb"
    if not osp.exists(epoch_dir) or not osp.exists(solution_path):
      break
    with open(solution_path, "rb") as file:
      solutions.append(cloudpickle.load(file))
  solution_path = path / "solution.pb"
  if osp.exists(solution_path):
    with open(solution_path, "rb") as file:
      solutions.append(cloudpickle.load(file))
  return solutions


def load_padded_solutions(path: pathlib.Path) -> list[strategy.Profile]:
  """Load the solutions from PSRO padded with zeros to the same shape.

  Args:
    path: Result directory from a run of PSRO.

  Returns:
    List of the solutions computed in the run.
  """
  return pad_solutions(load_game_matrix(path), load_solutions(path))


def load_game_matrix(path: pathlib.Path) -> np.ndarray:
  """Load the largest version of the game matrix saved.

  Args:
    path: Result directory from a run of PSRO.

  Returns:
    Game matrix from the result directory.
  """
  path = pathlib.Path(path)

  def _try_dir(p: pathlib.Path):
    """Try loading a game matrix from a directory."""
    p = p / "game_matrix.pb"
    if not osp.exists(p):
      return None
    with open(p, "rb") as file:
      return cloudpickle.load(file)

  if (result := _try_dir(path)) is not None:
    return result
  for epoch in range(load_num_epochs(path), -1, -1):
    if (result := _try_dir(path / f"epoch_{epoch}")) is not None:
      return result
  raise ValueError(f"No game matrix found in {path=}")


def pad_solutions(
    game_matrix: np.ndarray, solutions: strategy.Profile | list[strategy.Profile]
) -> list[strategy.Profile]:
  """Pad solutions to match a larger game matrix.

  Args:
    game_matrix: Game matrix.
    solutions: List of solutions on subsets of the game matrix.

  Returns:
    Solutions appended with zeros to match the dimensions of the game matrix.
  """
  player_to_len = dict(enumerate(game_matrix.shape[:-1]))

  def _pad(soln: strategy.Profile) -> strategy.Profile:
    """Pad a single profile."""
    return {p: np.append(soln[p], np.zeros(l - len(soln[p]))) for p, l in player_to_len.items()}

  if isinstance(solutions, list):
    new_solns = []
    for soln in solutions:
      new_solns.append(_pad(soln))
    return new_solns
  else:
    return _pad(solutions)
