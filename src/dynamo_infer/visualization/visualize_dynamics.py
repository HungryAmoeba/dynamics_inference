from . import create_visualizer
import numpy as np
import os
import datetime


def visualize_dynamics(config, pos, ori=None, save_path=None):
    """
    Visualize the dynamics of particles using the selected backend.
    Parameters:
    - config: Configuration object that contains visualization settings (must specify backend or method)
    - pos: Array of shape (T, N, D) representing positions of particles over time.
    - ori: Optional array of shape (T, N, D) representing orientations of particles over time.
    - save_path: Optional path to save the animation or rendering.
    Returns:
    None
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    visualizer = create_visualizer(config)
    return visualizer.visualize(pos, ori=ori, save_path=save_path)
