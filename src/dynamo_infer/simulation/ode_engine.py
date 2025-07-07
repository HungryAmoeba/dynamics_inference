"""ODE simulation engine using diffrax."""

import jax.numpy as jnp
from typing import Tuple
from .base import Simulator
from ..config.schemas import SimulationConfig
from ..dynamics.base import DynamicalSystem


class ODESimulator(Simulator):
    """ODE simulator using diffrax for robust integration."""
    
    def __init__(self, system: DynamicalSystem, config: SimulationConfig):
        super().__init__(system, config)
    
    def run(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the ODE simulation.
        
        Returns:
            (trajectory, times) where:
            - trajectory: (T, state_dim) array of states over time  
            - times: (T,) array of time points
        """
        print(f"Running ODE simulation with {self.config.solver}")
        print(f"Time: {self.config.time.t0} to {self.config.time.t1}, dt={self.config.time.dt}")
        
        # Placeholder implementation - would use diffrax in real version
        n_steps = int((self.config.time.t1 - self.config.time.t0) / self.config.time.dt)
        times = jnp.linspace(self.config.time.t0, self.config.time.t1, n_steps)
        
        # Get initial state
        initial_state = self.system.return_state()
        state_dim = initial_state.shape[0]
        
        # Dummy trajectory - would be actual integration in real version
        trajectory = jnp.zeros((n_steps, state_dim))
        trajectory = trajectory.at[0].set(initial_state)
        
        # Store results
        self.results = (trajectory, times)
        
        return trajectory, times