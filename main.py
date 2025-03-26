import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import subprocess
import tempfile
import os
from pathlib import Path

from dynamics.swarmalator import Swarmalator
from simulation.ode_solve_engine import ODEEngine
from visualizer.matplotlib_visualization import animate_particle_motion
from visualizer.blender_visualization import animate_particle_motion_blender

@hydra.main(version_base = None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    swarm = Swarmalator()
    swarm.initialize(cfg.dynamics.swarmalator) 

    engine = ODEEngine(swarm, cfg.engine.ode_solver)
    ys, ts = engine.run()
    # separate the positions and orientations
    # ys is of shape (T, 2ND) where N is the number of agents and D is the dimension
    pos = ys[:, :swarm.N * swarm.dim].reshape(-1, swarm.N, swarm.dim)
    ori = ys[:, swarm.N * swarm.dim:].reshape(-1, swarm.N, swarm.dim)

    # next check that ori always lies on the unit sphere
    norms_oris = jnp.linalg.norm(ori, axis=-1)
    # print max deviation of norms from 1
    print(f'Orientation normalization max deviation: {jnp.max(jnp.abs(norms_oris - 1))}')

    # # animate the particle motion
    # anim = animate_particle_motion(np.array(pos), np.array(ori))

    # fps = int(1 / cfg.engine.ode_solver.saveat.dt)
    # anim.save("swarmalator_motion.mp4", writer="ffmpeg", fps=fps, 
    #         extra_args=['-vcodec', 'libx264'])
    
    animate_particle_motion_blender(np.array(pos), np.array(ori))

    # # Blender visualization 
    # blender_vis = 0
    # if blender_vis:
    #     with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
    #         np.savez(tmp, pos=pos, ori=ori)
    #         tmp_path = tmp.name
    #         print(f"Saved temp data to: {tmp_path}")  # Debug path

    #     blender_exec = "/Applications/Blender.app/Contents/MacOS/Blender"
    #     script_path = Path(__file__).parent / "visualizer/blender_particle_visualization.py"
    #     # Run with error capture
    #     result = subprocess.run(
    #         [
    #             blender_exec,
    #             "--background",
    #             "--python", str(script_path),
    #             "--", tmp_path
    #         ],
    #         capture_output=True,
    #         text=True
    #     )
        
    #     # Write logs
    #     debug_log = Path.home()/"blender_swarmalator.log"
    #     with open(debug_log, 'a') as f:
    #         f.write("\nBlender Output:\n")
    #         f.write(result.stdout)
    #         f.write("\nBlender Errors:\n")
    #         f.write(result.stderr)
        
    #     print(f"Debug log saved to: {debug_log}")
    #     os.unlink(tmp_path)
    
        
if __name__ == "__main__":
    main()


def zap_small(arr, tol=1e-6):
    return jnp.where(jnp.abs(arr) < tol, 0, arr)
    # # first evaluate the performance of derivatives:
    # num_test = 100
    # for ind in range(num_test):
    #     # generate random state
    #     state = jrandom.normal(jrandom.PRNGKey(ind), (2 * swarm.N, swarm.dim))
    #     # compute the derivatives
    #     deriv_1 = swarm.compute_derivatives_unreshaped(state)
    #     deriv_2 = swarm.compute_derivatives_unreshaped_naive(state)
    #     diff = jnp.abs(deriv_1 - deriv_2)
    #     diff = zap_small(diff)

    #     # check if the derivatives are the same
        
    #     assert jnp.allclose(deriv_1, deriv_2, atol = 1e-5) == True, 'Derivatives are not the same for state: {}'.format(ind)

    # print('derivative computation is correct')
        
