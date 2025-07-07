"""Base classes for simulation engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import jax.numpy as jnp
from ..config.schemas import SimulationConfig
from ..dynamics.base import DynamicalSystem


class Simulator(ABC):
    """Abstract base class for simulation engines."""
    
    def __init__(self, system: DynamicalSystem, config: SimulationConfig):
        self.system = system
        self.config = config
        self.results = None
        
    @abstractmethod
    def run(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the simulation.
        
        Returns:
            (trajectory, times) where:
            - trajectory: (T, ...) array of states over time
            - times: (T,) array of time points
        """
        pass
    
    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get information about the simulation results."""
        if self.results is None:
            return {"completed": False}
            
        trajectory, times = self.results
        return {
            "completed": True,
            "n_timesteps": len(times),
            "time_span": (float(times[0]), float(times[-1])),
            "trajectory_shape": trajectory.shape,
        }