from gadynamics.dynamics.gravitation import GravitationalSystem
from gadynamics.dynamics.swarmalator import Swarmalator
from gadynamics.dynamics.interacting_GA import InteractingGA


def GetDynamicalSystem(config):
    """
    Factory function to get the appropriate dynamical system based on the provided configuration.

    Args:
        config: Configuration object that specifies the type of dynamical system.

    Returns:
        An instance of the specified dynamical system.
    """
    if config.type == "swarmalator":
        system = Swarmalator()
    elif config.type == "gravitation":
        system = GravitationalSystem()
    elif config.type == "ga_general":
        system = InteractingGA(equation_type=config.equation_type)
    else:
        raise ValueError(f"Unknown dynamical system type: {config.type}")

    system.initialize(config)

    return system
