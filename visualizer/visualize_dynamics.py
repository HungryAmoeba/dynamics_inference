from .matplotlib_visualization import animate_particle_motion
from .blender_visualization import animate_particle_motion_blender

import numpy as np
import os
import datetime


def visualize_dynamics(cfg, pos, ori=None, save_path=None):
    """
    Visualize the dynamics of particles using either Matplotlib or Blender.
    Parameters:
    - cfg: Configuration object that contains visualization settings.
    - pos: Array of shape (T, N, D) representing positions of particles over time.
    - ori: Optional array of shape (T, N, D) representing orientations of particles over time.

    Returns:
    None
    """
    # make numpy arrays if needed
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    if cfg.method == "matplotlib":
        anim = animate_particle_motion(pos, ori=ori, **cfg)

        # Save the animation to a file if specified in the config
        if save_path is None:
            base_dir = "./animation_outputs"
            random_file_name = (
                f"animation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            save_path = os.path.join(base_dir, random_file_name)

        fps = int(1 / cfg.interval)
        anim.save(
            save_path, writer="ffmpeg", fps=fps, extra_args=["-vcodec", "libx264"]
        )

    elif cfg.method == "blender":
        # Call the Blender animation function
        animate_particle_motion_blender(pos, ori=ori)

    else:
        raise ValueError(
            f"Visualization method '{cfg.visualizer.method}' is not supported. "
            "Please choose 'matplotlib' or 'blender'."
        )
