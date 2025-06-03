import numpy as np
from pathlib import Path


def report_trajectory_statistics(traj1, traj2, g_of_d, save_path=None):
    """
    Compute mean squared error (MSE) for overall and each grade, and optionally save to file.

    Parameters:
    - traj1, traj2: Arrays of shape (T, N, D)
    - g_of_d: List[int] of length D
    - save_path: Optional path to write the summary
    """
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    g_of_d = np.array(g_of_d)

    mse_total = np.mean((traj1 - traj2) ** 2)
    summary_lines = [f"Overall MSE: {mse_total:.6f}\n"]

    for g in sorted(set(g_of_d)):
        idxs = np.where(g_of_d == g)[0]
        if len(idxs) > 0:
            mse = np.mean((traj1[..., idxs] - traj2[..., idxs]) ** 2)
            summary_lines.append(f"Grade {g} MSE: {mse:.6f}\n")

    summary = "".join(summary_lines)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(summary)
    else:
        print(summary)
