import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
from pathlib import Path

from gadynamics.dynamics.system import GetDynamicalSystem
from gadynamics.simulation.ode_solve_engine import ODEEngine
from gadynamics.visualizer.visualize_dynamics import visualize_dynamics


def zap_small(arr, tol=1e-6):
    return jnp.where(jnp.abs(arr) < tol, 0, arr)


def get_save_name(config):
    dynamics_type = config.dynamics.type
    # use the datetime to create a unique name
    from datetime import datetime

    now = datetime.now()

    return f"{dynamics_type}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def save_results(save_path, ts, pos, ori):
    import os
    import numpy as np

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the results to a .npz file
    np.savez(save_path, ts=ts, pos=pos, ori=ori)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    system = GetDynamicalSystem(
        cfg.dynamics
    )  # Initialize the DynamicalSystem with the provided configuration

    engine = ODEEngine(system, cfg.engine.ode_solver)
    ys, ts = engine.run()

    # system.compute_derivatives(0, system.state, 0)

    # separate the positions and orientations
    # ys is of shape (T, 2ND) where N is the number of agents and D is the dimension
    pos, ori = system.unwrap_state(
        ys
    )  # Unwrap the state into positions and orientations
    visualize_dynamics(cfg.visualizer, pos, ori)

    # Save the results
    data_dir = Path(cfg.data_dir_base) / get_save_name(cfg)
    save_results(data_dir, ts, pos, ori)
    import pdb

    pdb.set_trace()
    system.print()


if __name__ == "__main__":
    main()
