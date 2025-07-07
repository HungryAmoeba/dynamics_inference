"""Factory functions for creating simulators."""

from typing import Dict, Type
from .base import Simulator
from .ode_engine import ODESimulator
from ..config.schemas import SimulationConfig
from ..dynamics.base import DynamicalSystem


# Registry of available simulators
SIMULATOR_REGISTRY: Dict[str, Type[Simulator]] = {
    "ode": ODESimulator,
    "diffrax": ODESimulator,  # Alias
}


def create_simulator(system: DynamicalSystem, config: SimulationConfig) -> Simulator:
    """
    Create a simulator from configuration.
    
    Args:
        system: Dynamical system to simulate
        config: Simulation configuration
        
    Returns:
        Configured simulator
        
    Raises:
        ValueError: If simulator type is not recognized
    """
    simulator_type = config.solver.lower()
    
    # Map solver names to simulator types
    if simulator_type in ["tsit5", "dopri5", "euler", "rk4"]:
        simulator_type = "ode"
    
    if simulator_type not in SIMULATOR_REGISTRY:
        available_types = list(SIMULATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown simulator type: {simulator_type}. "
            f"Available types: {available_types}"
        )
    
    simulator_class = SIMULATOR_REGISTRY[simulator_type]
    return simulator_class(system, config)