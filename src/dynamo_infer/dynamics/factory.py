"""Factory functions for creating dynamical systems."""

from typing import Dict, Type
from .base import DynamicalSystem
from ..config.schemas import DynamicsConfig

# Import system implementations
from .systems.swarmalator import Swarmalator
from .systems.gravitation import GravitationalSystem
from .systems.interacting_ga import InteractingGA
from .systems.swarmalator_breathing import SwarmalatorBreathing
from .systems.lattice_hamiltonian import LatticeHamiltonianSystem


# Registry of available systems
SYSTEM_REGISTRY: Dict[str, Type[DynamicalSystem]] = {
    "swarmalator": Swarmalator,
    "swarmalator_breathing": SwarmalatorBreathing,
    "gravitation": GravitationalSystem,
    "ga_general": InteractingGA,
    "lattice_hamiltonian": LatticeHamiltonianSystem,
}


def create_system(config: DynamicsConfig) -> DynamicalSystem:
    """
    Create a dynamical system from configuration.

    Args:
        config: Configuration specifying the system type and parameters

    Returns:
        Initialized dynamical system

    Raises:
        ValueError: If system type is not recognized
    """
    system_type = config.type

    if system_type not in SYSTEM_REGISTRY:
        available_types = list(SYSTEM_REGISTRY.keys())
        raise ValueError(
            f"Unknown dynamical system type: {system_type}. "
            f"Available types: {available_types}"
        )

    # Create system instance
    system_class = SYSTEM_REGISTRY[system_type]
    system = system_class()

    # Initialize with configuration
    system.initialize(config)

    return system


def register_system(name: str, system_class: Type[DynamicalSystem]) -> None:
    """
    Register a new dynamical system type.

    Args:
        name: Name to register the system under
        system_class: DynamicalSystem class to register
    """
    if not issubclass(system_class, DynamicalSystem):
        raise ValueError("system_class must be a subclass of DynamicalSystem")

    SYSTEM_REGISTRY[name] = system_class


def list_available_systems() -> Dict[str, Type[DynamicalSystem]]:
    """
    Get a dictionary of all available system types.

    Returns:
        Dictionary mapping system names to classes
    """
    return SYSTEM_REGISTRY.copy()
