import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from pathlib import Path


def compare_trajectories(
    traj1,
    traj2,
    g_of_d,
    num_objects=3,
    all_grades=False,
    labels=None,
    title=None,
    save_path=None,
):
    """
    Compare two trajectories by plotting them on the same figure and optionally saving to disk.

    Parameters:
    - traj1, traj2: Arrays of shape (T, N, D)
    - g_of_d: List[int], grade per D
    - num_objects: Number of agents/entities to visualize
    - all_grades: Plot all D components or one per grade
    - labels: Optional legend labels
    - title: Optional title for the whole figure
    - save_path: Path to save the figure, or None to just show
    """
    T, N, D = traj1.shape
    assert traj2.shape == traj1.shape, "Trajectory shapes must match"

    traj1 = np.array(traj1)
    traj2 = np.array(traj2)
    g_of_d = np.array(g_of_d)
    labels = labels or ["True", "Predicted"]

    unique_grades = np.unique(g_of_d)
    indices_to_plot = (
        np.arange(D)
        if all_grades
        else [np.where(g_of_d == g)[0][0] for g in unique_grades]
    )

    np.random.seed(0)
    selected_entities = np.random.choice(N, min(num_objects, N), replace=False)

    n_rows = len(selected_entities)
    n_cols = len(indices_to_plot)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )

    for i, n in enumerate(selected_entities):
        for j, d in enumerate(indices_to_plot):
            ax = axes[i][j]
            ax.plot(traj1[:, n, d], label=labels[0], linewidth=2)
            ax.plot(traj2[:, n, d], label=labels[1], linestyle="--")
            ax.set_title(f"Obj {n}, Grade {g_of_d[d]}, Index {d}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            if i == 0 and j == 0:
                ax.legend()

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
