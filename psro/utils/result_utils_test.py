"""Test suite for `load_solutions` function."""
import pathlib
import tempfile
from typing import Any
from unittest import mock

import cloudpickle
import numpy as np
from absl.testing import absltest, parameterized

from psro import strategy
from psro.utils import result_utils


class LoadSolutionsTest(parameterized.TestCase):
  """Test suite for `load_solutions` function."""

  def setUp(self):
    # Set up a temporary directory for the tests
    self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
    self.addCleanup(self.test_dir.cleanup)

  def create_solution_files(self, num_epochs: int, dir_path: pathlib.Path):
    """Helper method to create mock solution files for testing."""
    for epoch in range(1, num_epochs + 1):
      epoch_dir = dir_path / f"epoch_{epoch}"
      epoch_dir.mkdir(parents=True, exist_ok=True)
      solution_path = epoch_dir / "solution.pb"
      with open(solution_path, "wb") as file:
        cloudpickle.dump(f"Solution for epoch {epoch}", file)

  @parameterized.parameters((1,), (3,), (5,))
  def test_load_solutions_with_multiple_epochs(self, num_epochs: int):
    """Tests if the correct number of solutions are loaded for multiple epochs."""
    dir_path = pathlib.Path(self.test_dir.name)
    self.create_solution_files(num_epochs, dir_path)
    with mock.patch("cloudpickle.load", side_effect=lambda file: file.read()):
      solutions = result_utils.load_solutions(dir_path)
      self.assertEqual(len(solutions), num_epochs, "Should load all epochs solutions")

  def test_load_solutions_with_nonexistent_path(self):
    """Tests if the function handles non-existent paths correctly."""
    dir_path = pathlib.Path(self.test_dir.name) / "nonexistent"
    solutions = result_utils.load_solutions(dir_path)
    self.assertEqual(solutions, [], "Should return an empty list for non-existent paths")

  def test_load_solutions_without_epochs(self):
    """Tests if the function can handle a directory without epoch subdirectories."""
    dir_path = pathlib.Path(self.test_dir.name)
    solutions = result_utils.load_solutions(dir_path)
    self.assertEqual(solutions, [], "Should return an empty list if there are no epoch directories")

  def test_load_solutions_with_final_solution(self):
    """Tests if the function loads the final solution saved at the root of the result directory."""
    dir_path = pathlib.Path(self.test_dir.name)
    self.create_solution_files(3, dir_path)
    # Create an additional solution file
    solution_path = dir_path / "solution.pb"
    with open(solution_path, "wb") as file:
      cloudpickle.dump("Final Solution", file)

    with mock.patch("cloudpickle.load", side_effect=lambda file: file.read()):
      solutions = result_utils.load_solutions(dir_path)
      self.assertEqual(len(solutions), 4, "Should include the additional 'solution.pb' not in an epoch directory")


class LoadNumEpochsTest(parameterized.TestCase):
  """Test suite for `load_num_epochs` function."""

  def setUp(self):
    # Set up a temporary directory for the tests
    self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
    self.addCleanup(self.test_dir.cleanup)

  def create_epoch_files(self, num_epochs, valid=True):
    """Helper method to create mock epoch directories and files for testing."""
    dir_path = pathlib.Path(self.test_dir.name)
    for epoch in range(1, num_epochs + 1):
      epoch_dir = dir_path / f"epoch_{epoch}"
      epoch_dir.mkdir(parents=True, exist_ok=True)
      # Create required files only if the epoch should be valid.
      if valid:
        (epoch_dir / "game_matrix.pb").touch()
        (epoch_dir / "solution.pb").touch()

  @parameterized.parameters(
      (3, True), (5, True), (7, False)  # This will test the scenario where the epoch directories are invalid
  )
  def test_get_largest_epoch_number_with_files(self, num_epochs, valid):
    """Tests if the correct largest epoch number is returned."""
    self.create_epoch_files(num_epochs, valid)
    expected_largest_epoch = num_epochs if valid else 0
    largest_epoch = result_utils.load_num_epochs(self.test_dir.name)
    self.assertEqual(
        largest_epoch,
        expected_largest_epoch,
        f"Expected largest epoch to be {expected_largest_epoch}, but got {largest_epoch}",
    )


class LoadGameMatrixTest(parameterized.TestCase):
  """Test suite for `load_game_matrix` function."""

  def setUp(self):
    # Set up a temporary directory for the tests
    self.test_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
    self.addCleanup(self.test_dir.cleanup)

  def create_test(self, epoch: None | pathlib.Path, content: Any):
    """Helper method to create a mock game matrix file for testing."""
    if epoch is not None:
      for i in range(1, epoch + 1):
        epoch_dir = pathlib.Path(self.test_dir.name) / f"epoch_{i}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        (epoch_dir / "solution.pb").touch()
        file_path = epoch_dir / "game_matrix.pb"
        file_path.touch()
    else:
      file_path = pathlib.Path(self.test_dir.name) / "game_matrix.pb"
    with open(file_path, "wb") as file:
      cloudpickle.dump(content, file)

  @parameterized.parameters(
      (None, "Game Matrix Content"),
      (5, "Game Matrix Content"),
      (3, "Game Matrix Content"),
  )
  def test_load_game_matrix(self, epoch, content):
    """Tests if the game matrix is correctly loaded from the largest epoch or main directory."""
    self.create_test(epoch, content)
    game_matrix = result_utils.load_game_matrix(pathlib.Path(self.test_dir.name))
    self.assertEqual(game_matrix, content)


class PadSolutionsTest(parameterized.TestCase):
  """Test suite for `pad_solutions` function."""

  @parameterized.parameters(
      {
          "game_matrix": np.zeros((3, 3, 2)),
          "solutions": {0: np.array([1, 1]), 1: np.array([1])},
          "padded_solutions": {0: np.array([1, 1, 0]), 1: np.array([1, 0, 0])},
      },
      {
          "game_matrix": np.zeros((3, 3, 2)),
          "solutions": [{0: np.array([1, 1]), 1: np.array([1])}],
          "padded_solutions": [{0: np.array([1, 1, 0]), 1: np.array([1, 0, 0])}],
      },
      {
          "game_matrix": np.zeros((3, 3, 2)),
          "solutions": [
              {0: np.array([1]), 1: np.array([1])},
              {0: np.array([0, 0.5]), 1: np.array([0, 0.5])},
              {0: np.array([1, 1, 1]), 1: np.array([1, 1, 1])},
          ],
          "padded_solutions": [
              {0: np.array([1, 0, 0]), 1: np.array([1, 0, 0])},
              {0: np.array([0, 0.5, 0]), 1: np.array([0, 0.5, 0])},
              {0: np.array([1, 1, 1]), 1: np.array([1, 1, 1])},
          ],
      },
  )
  def test_pad_solutions(
      self,
      game_matrix: np.ndarray,
      solutions: strategy.Profile | list[strategy.Profile],
      padded_solutions: list[strategy.Profile],
  ):
    """Test `pad_solutions`."""
    result = result_utils.pad_solutions(game_matrix, solutions)
    if not isinstance(padded_solutions, list):
      result = [result]
      padded_solutions = [padded_solutions]

    for expected, actual in zip(padded_solutions, result):
      self.assertLen(actual, len(expected))
      for key, value in expected.items():
        np.testing.assert_almost_equal(value, actual[key])


if __name__ == "__main__":
  absltest.main()
