from .gravitation import GravitationalSystem
from .swarmalator import Swarmalator

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
    else:
        raise ValueError(f"Unknown dynamical system type: {config.type}")
    
    system.initialize(config)

    return system