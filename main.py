import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import subprocess
import tempfile
import os
from pathlib import Path

from dynamics.system import GetDynamicalSystem
from simulation.ode_solve_engine import ODEEngine
from visualizer.visualize_dynamics import visualize_dynamics

@hydra.main(version_base = None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    system = GetDynamicalSystem(cfg.dynamics)  # Initialize the DynamicalSystem with the provided configuration

    engine = ODEEngine(system, cfg.engine.ode_solver)
    ys, ts = engine.run()
    # separate the positions and orientations
    # ys is of shape (T, 2ND) where N is the number of agents and D is the dimension
    pos, ori = system.unwrap_state(ys)  # Unwrap the state into positions and orientations

    visualize_dynamics(cfg.visualizer, pos, ori)

    
        
if __name__ == "__main__":
    main()


def zap_small(arr, tol=1e-6):
    return jnp.where(jnp.abs(arr) < tol, 0, arr)
