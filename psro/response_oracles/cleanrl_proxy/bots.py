"""Bots to serve as proxies for CleanRL models."""
import dataclasses

import jax
import jax.numpy as jnp
from marl import individuals, types, worlds
from marl.services import snapshotter


@dataclasses.dataclass
class ValueProxyBot(individuals.Bot):
  """Base class for types that are able to interact in a world."""

  q_network_snapshot: snapshotter.Snapshot

  def __post_init__(self):
    """Post initializer."""
    self._init = False
    self.q_network = None

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> tuple[types.State, types.Action]:
    """Select an action to take given the current timestep."""
    values = self.q_network.apply(self.q_network_snapshot.params, timestep.observation)
    return state, jnp.argmax(values, axis=-1).tolist()

  def episode_reset(self, timestep: worlds.TimeStep):
    """Reset the agent's episodic state."""
    self._maybe_lazy_load()
    del timestep
    return ()

  def _maybe_lazy_load(self):
    """Lazy load the Q-Network."""
    if self._init:
      return

    self.q_network = self.q_network_snapshot.ctor(**self.q_network_snapshot.ctor_kwargs)
    self.q_network.apply = jax.jit(self.q_network.apply)
    self._init = True


@dataclasses.dataclass
class OpenSpielValueProxyBot(individuals.Bot):
  """Base class for types that are able to interact in a world."""

  q_network_snapshot: snapshotter.Snapshot

  def __post_init__(self):
    """Post initializer."""
    self._init = False
    self.q_network = None

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> tuple[types.State, types.Action]:
    """Select an action to take given the current timestep."""
    observation = timestep.observation["info_state"]
    values = self.q_network.apply(self.q_network_snapshot.params, observation)
    if "legal_actions" in timestep.observation:
      values = values.at[~timestep.observation["legal_actions"]].set(-jnp.inf)
    return state, jnp.argmax(values, axis=-1).tolist()

  def episode_reset(self, timestep: worlds.TimeStep):
    """Reset the agent's episodic state."""
    self._maybe_lazy_load()
    del timestep
    return ()

  def _maybe_lazy_load(self):
    """Lazy load the Q-Network."""
    if self._init:
      return

    self.q_network = self.q_network_snapshot.ctor(**self.q_network_snapshot.ctor_kwargs)
    self.q_network.apply = jax.jit(self.q_network.apply)
    self._init = True
