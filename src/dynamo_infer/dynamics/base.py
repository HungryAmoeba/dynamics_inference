"""Base classes for dynamical systems."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import jax.numpy as jnp
from ..config.schemas import DynamicsConfig


class DynamicalSystem(ABC):
    """Abstract base class for all dynamical systems."""
    
    def __init__(self):
        self.config: Optional[DynamicsConfig] = None
        self.state: Optional[jnp.ndarray] = None
        self.n_particles: int = 0
        self.dimension: int = 0
        self.initialized: bool = False
    
    @abstractmethod
    def initialize(self, config: DynamicsConfig) -> None:
        """
        Initialize the dynamical system with the provided configuration.
        
        Args:
            config: Configuration object specifying system parameters
        """
        self.config = config
        self.n_particles = config.n_particles
        self.dimension = config.dimension
        
    @abstractmethod
    def compute_derivatives(self, t: float, state: jnp.ndarray, args: Any = None) -> jnp.ndarray:
        """
        Compute the derivative of the system at the given state.
        
        Args:
            t: Current time
            state: Current state vector
            args: Additional arguments
            
        Returns:
            Time derivative of the state
        """
        pass
    
    @abstractmethod
    def return_state(self) -> jnp.ndarray:
        """
        Return the current state of the system.
        
        Returns:
            Current state vector
        """
        pass
    
    @abstractmethod
    def unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """
        Unwrap the state vector into its constituent parts (positions, orientations, etc.).
        
        Args:
            state: State vector to unwrap
            
        Returns:
            Tuple of state components (e.g., positions, orientations, volumes)
        """
        pass
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current state.
        
        Returns:
            Dictionary with state information
        """
        if self.state is None:
            return {"initialized": False}
            
        return {
            "initialized": self.initialized,
            "n_particles": self.n_particles,
            "dimension": self.dimension,
            "state_shape": self.state.shape,
            "state_norm": float(jnp.linalg.norm(self.state)),
        }
    
    def validate_state(self, state: jnp.ndarray) -> bool:
        """
        Validate that a state vector has the correct format.
        
        Args:
            state: State vector to validate
            
        Returns:
            True if state is valid
        """
        if not self.initialized:
            return False
            
        expected_shape = self.get_expected_state_shape()
        return state.shape == expected_shape
    
    @abstractmethod
    def get_expected_state_shape(self) -> Tuple[int, ...]:
        """
        Get the expected shape of the state vector.
        
        Returns:
            Expected state shape
        """
        pass
    
    def print_info(self) -> None:
        """Print information about the system."""
        print(f"System type: {self.__class__.__name__}")
        print(f"Initialized: {self.initialized}")
        if self.initialized:
            print(f"Particles: {self.n_particles}")
            print(f"Dimension: {self.dimension}")
            print(f"State shape: {self.get_expected_state_shape()}")


class InteractingParticleSystem(DynamicalSystem):
    """Base class for systems of interacting particles."""
    
    def __init__(self):
        super().__init__()
        self.positions: Optional[jnp.ndarray] = None
        self.orientations: Optional[jnp.ndarray] = None
        
    def unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Unwrap state into positions and orientations.
        
        Returns:
            (positions, orientations) where positions has shape (N, D) and 
            orientations has shape (N, D)
        """
        pos_size = self.n_particles * self.dimension
        positions = state[:pos_size].reshape(self.n_particles, self.dimension)
        orientations = state[pos_size:].reshape(self.n_particles, self.dimension)
        return positions, orientations
    
    def get_expected_state_shape(self) -> Tuple[int]:
        """Expected state shape for particle systems."""
        return (2 * self.n_particles * self.dimension,)