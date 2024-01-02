"""Policy-Space Response Oracles."""
import itertools
import multiprocessing
import os
import pathlib
import shutil
import traceback
from concurrent import futures
from typing import Any, Sequence

import cloudpickle
import numpy as np
from absl import logging
from marl import bots, types

from psro import core, empirical_games, strategy
from psro.utils import result_utils


class PSRO:
  """Policy-Space Response Oracles.

  Iteratively performs:
    1. Game reasoning.
    2. Strategy exploration.

  Args:
    game_ctor: Constructor for the _real_ game being solved.
    initial_strategies: Initial strategy sets for all players. There must be at least one policy
      in each player's strategy set.
    response_oracles:
    profile_simulator:
    game_solver:
    result_dir: Directory to write intermediate and final results.
    num_samples_per_profile: Number of return samples to simulate for each strategy profile.
    solution_precision: Precision of computed solutions.
    max_profile_simulator_workers:
    max_response_oracle_workers:
    initial_empirical_game: Initial state of the empirical game. Policy indices in this empirical
      game must match those in `initial_strategies`.
  """

  def __init__(
      self,
      game_ctor: core.GameCtor,
      initial_strategies: strategy.JointStrategy,
      response_oracles: dict[types.PlayerID, core.ResponseOracle],
      profile_simulator: core.ProfileSimulator,
      game_solver: core.GameSolver,
      result_dir: pathlib.Path | str,
      *,
      num_samples_per_profile: int = 30,
      solution_precision: int = 5,
      max_profile_simulator_workers: int = 1,
      max_response_oracle_workers: int = 1,
      initial_empirical_game: empirical_games.NormalForm | empirical_games.NormalFormSQL | None = None,
  ):
    """Initializer."""
    self._game_ctor = game_ctor
    self.num_players = len(initial_strategies)
    for player_id, player_strategy in initial_strategies.items():
      if len(player_strategy) < 1:
        raise ValueError(f"Player {player_id} not assigned an initial policy.")
    self._strategies = initial_strategies
    self._response_oracles = response_oracles
    self._profile_simulator = profile_simulator
    self._game_solver = game_solver
    self._result_dir = pathlib.Path(os.path.abspath(result_dir))
    self._num_samples_per_profile = num_samples_per_profile
    self._solution_precision = solution_precision
    self._max_profile_simulation_workers = max_profile_simulator_workers
    self._max_response_oracle_workers = max_response_oracle_workers

    if initial_empirical_game:
      self._empirical_game = initial_empirical_game
    else:
      self._empirical_game = empirical_games.NormalFormSQL(self.num_players, self._result_dir / "empirical_game.sql")

  def run(self, num_epochs: core.Epoch | None = None):
    """Run PSRO.

    Args:
        num_epochs: Optionally, limit the number of epochs that are run.
    """
    logging.info("Running PSRO.")
    if num_epochs is None:
      logging.info("No limit of epochs specified, running indefinitely.")
      num_epochs = np.iinfo(np.int32).max
    self.save_initial_strategies()

    for epoch_i in range(1, num_epochs + 1):
      logging.info("Beginning Epoch %d", epoch_i)
      epoch_dir = self._result_dir / f"epoch_{epoch_i}"
      os.mkdir(epoch_dir)

      self.simulate_profiles()
      solution = self.solve_empirical_game(epoch_dir)
      self.expand_empirical_game(epoch_dir, solution)
      self._maybe_symlink_policies(epoch_i)

    logging.info("Finalizing empirical game.")
    self.simulate_profiles()
    self.solve_empirical_game(self._result_dir)
    logging.info("PSRO finished.")

  def simulate_profiles(self):
    """Simulates all profiles.

    This method checks all strategy profiles to ensure that they have the minimum number of required
    samples. If not, schedules and executes simulations for each profile to complete the empirical game.
    """
    logging.info("Simulating profiles that are undersampled.")

    # Collect the strategies that need simulation, and the number of simulations required.
    total_samples_needed = 0
    needed_simulations = []

    pure_profiles = [np.arange(len(self._strategies[id])) for id in range(self.num_players)]
    pure_profiles = itertools.product(*pure_profiles)

    for pure_profile in pure_profiles:
      num_samples = self._empirical_game.num_samples(pure_profile)
      num_needed_samples = self._num_samples_per_profile - num_samples
      if num_needed_samples:
        needed_simulations.append(
            core.SimulationJob(
                game_ctor=self._game_ctor,
                players=self._strategies,
                profile=pure_profile,
                num_episodes=num_needed_samples,
            )
        )
        total_samples_needed += num_needed_samples
    logging.info("Found %d profiles requiring %d total samples.", len(needed_simulations), total_samples_needed)

    # Simulate the needed strategy profiles.
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=self._max_profile_simulation_workers, mp_context=context) as executor:
      for profile, payoffs in executor.map(self._profile_simulator, needed_simulations):
        self._empirical_game.add_payoffs(dict(enumerate(profile)), payoffs)
    logging.info("Simulation complete.")

  def solve_empirical_game(self, epoch_dir: pathlib.Path) -> strategy.MixedProfile:
    """Solve the current empirical game."""
    logging.info("Solving the empirical game.")
    matrix = self._empirical_game.game_matrix()
    solutions = self._game_solver(matrix)
    if isinstance(solutions, Sequence):
      logging.info("Found %d solutions, taking first solution.", len(solutions))
      solution = solutions[0]
    else:
      solution = solutions
    # solution = self._fix_solution_precision(solution)

    logging.info("Solution: %s", solution)
    with open(epoch_dir / "solution.pb", "wb") as file:
      cloudpickle.dump(solution, file)
    with open(epoch_dir / "game_matrix.pb", "wb") as file:
      cloudpickle.dump(matrix, file)

    for player_id, mixture in solution.items():
      self._strategies[player_id].mixture = mixture
    return solution

  def expand_empirical_game(self, epoch_dir: pathlib.Path, solution: strategy.MixedProfile):
    """Expands the game by having each player compute one new response policy.

    Args:
        epoch_dir: Directory in which to save artifacts generated during game expansion.
    """
    job_template = core.ResponseOracleJob(
        learner_id=None,
        players=self._strategies,
        game_ctor=self._game_ctor,
        solution=solution,
        epoch_dir=epoch_dir,
    )
    max_workers = min(self._max_response_oracle_workers, self.num_players)
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
      oracle_futures = []
      for learner_id, response_oracle in self._response_oracles.items():
        oracle_futures.append(executor.submit(response_oracle, job_template._replace(learner_id=learner_id)))
      for oracle_future in futures.as_completed(oracle_futures):
        try:
          learner_id, policy = oracle_future.result()
          self._strategies[learner_id].add_policy(policy)
        except Exception as e:
          print(f"Response oracle died with exception: {e}")
          traceback.print_exc()
          raise RuntimeError("Error encountered when running a response oracle.") from e

  def save_initial_strategies(self):
    """Save the initial policies to disk."""
    # Directory to save initial strategies.
    save_dir = self._result_dir / "initial_strategies/"
    os.mkdir(save_dir)

    # Directory to accumulate symlinks to all policies.
    policy_dir = self._result_dir / "policies"
    os.makedirs(policy_dir)

    for player, strat in self._strategies.items():
      player_dir = save_dir / f"player_{player}/"
      os.mkdir(player_dir)
      os.mkdir(policy_dir / f"player_{player}")

      for policy_i, policy in enumerate(strat.policies):
        policy_path = player_dir / f"policy_{policy_i}.pb"
        with open(policy_path, "wb") as file:
          cloudpickle.dump(policy, file)
        link_path = policy_dir / f"player_{player}" / f"policy_{policy_i}.pb"
        os.symlink(policy_path, link_path)

  @property
  def empirical_game(self) -> empirical_games.NormalForm:
    """Getter for the empirical game."""
    return self._empirical_game

  @property
  def strategies(self) -> strategy.JointStrategy:
    """Getter for the players's strategies."""
    return self._strategies

  def _fix_solution_precision(self, solution: strategy.MixedProfile) -> strategy.MixedProfile:
    """Set a solution to a specificed precision."""
    for player_id, mixture in solution.items():
      mixture = np.round(mixture, self._solution_precision)
      rounding_error = 1.0 - np.sum(mixture)
      highest_support = np.argmax(mixture)
      mixture[highest_support] += rounding_error
      solution[player_id] = mixture
    return solution

  def _maybe_symlink_policies(self, epoch: int):
    """Maintain symlinks at the run-level directory of all players' policies.

    This softly assumes that response oracles record policies in a standard structure:
      epoch_i/
        player_i/
          policy.pb
        player_j/
          policy.pb
        ...
      ...

    TODO(max): Don't repeat work across epochs. Currently does this because it
      allows this to be self-contained.
    """
    policy_dir = self._result_dir / "policies"

    for player in range(self.num_players):
      player_dir = policy_dir / f"player_{player}"

      files = (entry for entry in player_dir.iterdir() if entry.is_file())
      policy_id = sum(1 for _ in files)

      epoch_dir = self._result_dir / f"epoch_{epoch}"
      if not os.path.exists(epoch_dir):
        break
      # If a policy was saved by the response oracle, establish a symlink.
      # TODO(maxsmith): Allow multiple policies to be constructed in an epoch.
      policy_path = epoch_dir / f"player_{player}" / "policy.pb"
      link_path = player_dir / f"policy_{policy_id}.pb"
      if os.path.exists(policy_path) and not os.path.exists(link_path):
        os.symlink(policy_path, link_path)
      policy_id += 1


def load_partial_run(
    game_ctor: core.GameCtor,
    response_oracles: dict[types.PlayerID, core.ResponseOracle],
    profile_simulator: core.ProfileSimulator,
    game_solver: core.GameSolver,
    *,
    load_dir: pathlib.Path | str,
    continue_dir: pathlib.Path | str,
    **psro_kwargs: dict[str, Any],
) -> PSRO:
  """Load a partial run of PSRO to be continued as a new run.

  NOTE: This method forks the entire result directory to prevent contamination.

  TODO(max): Write flags/settings to disk at the start of PSRO run. Then we can
    directly load them here instead of requesting PSRO parameters and assuming
    that they are consistent.
  """
  load_dir = pathlib.Path(load_dir)
  continue_dir = pathlib.Path(continue_dir)
  dummy_strat = strategy.Strategy([bots.RandomActionBot(2)], mixture=[1.0])

  psro = PSRO(
      game_ctor=game_ctor,
      # PSRO requires an initial strategy set. We set it for comptability, and then
      # will overwrite all of the associated internal states and disk artifacts.
      initial_strategies={player: dummy_strat for player in response_oracles.keys()},
      response_oracles=response_oracles,
      profile_simulator=profile_simulator,
      game_solver=game_solver,
      result_dir=continue_dir,
      **psro_kwargs,
  )

  _copy_directory_contents(load_dir, continue_dir)
  # pylint: disable=protected-access
  psro._strategies = result_utils.load_joint_strategy(continue_dir)
  psro._empirical_game = result_utils.load_empirical_game(continue_dir, len(psro._strategies))
  # pylint: enable=protected-access

  return psro, result_utils.load_num_epochs(continue_dir)


def _copy_directory_contents(src, dst):
  """Copy a directories contents."""
  if not os.path.exists(dst):
    os.makedirs(dst)
  for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
      _copy_directory_contents(s, d)
    else:
      shutil.copy2(s, d)
