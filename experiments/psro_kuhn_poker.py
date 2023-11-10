"""Compute a solution to kuhn poker using PSRO."""
import os
import os.path as osp
import pathlib
import shutil

import pyspiel
from absl import app

from marl import bots
from marl.games import openspiel_proxy
from psro import strategy
from psro._psro import PSRO
from psro.profile_simulators import marl_bot_simulation
from psro.response_oracles import openspiel_br
from psro.solvers import gambit_solver


def game_ctor():
  """Constructs an instance of kuhn Poker."""
  game = pyspiel.load_game("kuhn_poker")
  return openspiel_proxy.OpenSpielProxy(game, include_full_state=True)


def main(argv):
  """."""
  result_dir = pathlib.Path("test_run/")
  if osp.exists(result_dir):
    shutil.rmtree(result_dir)
  os.mkdir(result_dir)

  PSRO(
      game_ctor=game_ctor,
      initial_strategies={
          0: strategy.Strategy([bots.RandomActionBot(2)], mixture=[1.0]),
          1: strategy.Strategy([bots.RandomActionBot(2)], mixture=[1.0]),
      },
      response_oracles={
          0: openspiel_br.BestResponse(pyspiel.load_game("kuhn_poker")),
          1: openspiel_br.BestResponse(pyspiel.load_game("kuhn_poker")),
      },
      profile_simulator=marl_bot_simulation.simulate_profile,
      game_solver=gambit_solver.GambitSolver(),
      result_dir=result_dir,
  ).run(num_epochs=5)


if __name__ == "__main__":
  app.run(main)
