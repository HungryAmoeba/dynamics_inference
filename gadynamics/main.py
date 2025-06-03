import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
from pathlib import Path

from gadynamics.dynamics.system import GetDynamicalSystem
from gadynamics.simulation.ode_solve_engine import ODEEngine
from gadynamics.visualizer.visualize_dynamics import visualize_dynamics
from gadynamics.inference.GA_inference import GA_DynamicsInference
from gadynamics.inference.feature_library.orthogonal_poly_library import (
    OrthogonalPolynomialLibrary,
)
from gadynamics.inference.utils.to_GA import pos_ori_to_GA

from gadynamics.inference.evaluation.make_figures import compare_trajectories
from gadynamics.inference.evaluation.get_metrics import report_trajectory_statistics


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
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    system = GetDynamicalSystem(
        cfg.dynamics
    )  # Initialize the DynamicalSystem with the provided configuration

    # # do one step of the system to help debug
    # y0 = system.return_state()
    # deriv = system.compute_derivatives(0, y0, 0)
    # import pdb; pdb.set_trace()

    engine = ODEEngine(system, cfg.engine.ode_solver)
    ys, ts = engine.run()

    # system.compute_derivatives(0, system.state, 0)

    # necessary for swarmalator and gravitation I think? And visualization
    # # separate the positions and orientations

    pos, ori, vol = system.unwrap_state(
        ys
    )  # Unwrap the state into positions and orientations

    # visualize_dynamics(cfg.visualizer, pos, ori)

    # Save the results
    data_dir = Path(cfg.data_dir_base) / get_save_name(cfg)
    # save_results(data_dir, ts, pos, ori)

    # system.print()

    # now run inference on the system
    # convert the positions and orientations to GA format
    GA_trajectory = pos_ori_to_GA(pos, ori, t=ts, vol=vol)

    # assume ys is already in GA format
    model = GA_DynamicsInference(feature_library=OrthogonalPolynomialLibrary(degree=3))
    model.fit(GA_trajectory, epochs=int(1e5), print_every=10000)

    # TODO: clean this up!
    preds_derivs = model.predict(GA_trajectory)
    _, deriv_gt = model._process_trajectories(GA_trajectory, None, None)

    # compare preds_derivs, preds_gt
    fig_path = output_dir / "figures" / "trajectory_comparison.png"
    txt_path = output_dir / "results" / "mse_report.txt"

    model_save = output_dir / "results" / "model.txt"

    if not model_save.parent.exists():
        model_save.parent.mkdir(parents=True)

    model.generate_latex_eq(model_save)

    # compare_trajectories(deriv_gt, preds_derivs, model.g_of_d, num_objects=3, all_grades=False, title="Model vs Ground Truth")

    compare_trajectories(
        deriv_gt,
        preds_derivs,
        model.g_of_d,
        save_path=fig_path,
        title="Inference Model vs Ground Truth",
    )

    if not txt_path.parent.exists():
        txt_path.parent.mkdir(parents=True)

    report_trajectory_statistics(
        deriv_gt, preds_derivs, model.g_of_d, save_path=txt_path
    )


if __name__ == "__main__":
    main()
