import bpy
import numpy as np
import tempfile
import subprocess
import sys
from pathlib import Path

def animate_particle_motion_blender(pos: np.ndarray, ori: np.ndarray):
    """
    Renders particle animation in Blender without leaving .npy files.
    
    Args:
        pos: [T, N, 3] position array
        ori: [T, N, 4] orientation array (quaternions)
    """
    if ori.shape[-1] == 3:
        # this is just a vector on the unit sphere
        # convert to quaternions 
        ori = np.concatenate([np.zeros(ori.shape[:-1] + (1,)), ori], axis=-1)

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix="_pos.npy", delete=False) as pos_file, \
         tempfile.NamedTemporaryFile(suffix="_ori.npy", delete=False) as ori_file:
         
        np.save(pos_file, pos)
        np.save(ori_file, ori)
        temp_paths = [pos_file.name, ori_file.name]

    # Get paths
    blender_script = str(Path(__file__).parent / "blender_script.py")
    blender_exec = find_blender_executable()  # Implement platform detection

    # Launch Blender
    subprocess.run([
        blender_exec,
        "--background",  # Run without UI
        "--python", blender_script,
        "--",  # End of Blender arguments
        *temp_paths
    ], check=True)

    # Cleanup temp files even if Blender crashes
    for path in temp_paths:
        Path(path).unlink(missing_ok=True)

def find_blender_executable():
    """Platform-specific Blender executable finder"""
    # Implement for your OS (example for Linux/Mac/Windows)
    if sys.platform == "linux":
        return "/usr/bin/blender"
    elif sys.platform == "darwin":
        return "/Applications/Blender.app/Contents/MacOS/Blender"
    elif sys.platform == "win32":
        return "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe"
    else:
        raise OSError("Unsupported OS")
    