import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation


def animate_temporal_graph(
    pos, graph, ori=None, node_colors=None, edge_colors=None, interval=200, **kwargs
):
    """
    Animate a temporal graph with node positions and optional orientations.

    Args:
        pos: Array of shape (T, N, D) where T is time, N is number of nodes, D is dimensions (2 or 3).
        graph: NetworkX graph object representing the structure of the graph.
        ori: Optional array of shape (T, N, D) for node orientations.
        node_colors: Optional array of shape (N,) or (T, N) for dynamic node colors.
        edge_colors: Optional list of edge colors (static or dynamic).
        interval: Milliseconds between frames.
        kwargs: Additional arguments like title.
    """
    T, N, D = pos.shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D graph animations supported.")

    # Convert to numpy if JAX arrays
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)
    if node_colors is not None and hasattr(node_colors, "device_buffer"):
        node_colors = np.array(node_colors)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d" if D == 3 else None)

    # Set limits
    all_pos = pos.reshape(-1, D)
    min_vals = all_pos.min(axis=0)
    max_vals = all_pos.max(axis=0)
    padding = 0.1 * (max_vals - min_vals)

    if D == 3:
        ax.set_xlim3d(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim3d(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim3d(min_vals[2] - padding[2], max_vals[2] + padding[2])

        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])
        if "zlabel" in kwargs:
            ax.set_zlabel(kwargs["zlabel"])
    else:
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])

        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])

    # Initialize node scatter
    initial_colors = (
        node_colors[0]
        if node_colors is not None and node_colors.ndim == 2
        else node_colors
    )
    if D == 2:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], c=initial_colors, s=100, zorder=2)
    else:
        scat = ax.scatter(
            pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], c=initial_colors, s=100
        )

    # Initialize orientation arrows
    # precompute arrow length = half the minimum nonzero distance at t=0
    p0 = pos[0]  # (N,D)
    # compute pairwise squared distances
    diffs = p0[:, None, :] - p0[None, :, :]  # (N,N,D)
    d2 = np.sum(diffs**2, axis=-1)  # (N,N)
    d2[np.eye(N, dtype=bool)] = np.inf  # mask diagonal
    min_d2 = d2.min()  # smallest nonzero squared dist
    arrow_length = 0.5 * np.sqrt(min_d2)

    arrow_scale = arrow_length if ori is not None else None
    quivers = None
    if ori is not None and D == 2:
        quivers = ax.quiver(
            pos[0, :, 0],
            pos[0, :, 1],
            ori[0, :, 0],
            ori[0, :, 1],
            color="red",
            scale_units="xy",
            scale=1 / arrow_scale,
            zorder=1,
        )

    # Draw static edge list once
    edge_lines = []
    for i, j in graph.edges:
        if D == 3:
            line = ax.plot(
                [pos[0, i, 0], pos[0, j, 0]],
                [pos[0, i, 1], pos[0, j, 1]],
                [pos[0, i, 2], pos[0, j, 2]],
                c=edge_colors[i] if edge_colors is not None else "gray",
                lw=1.5,
                alpha=0.6,
            )[0]
        else:
            line = ax.plot(
                [pos[0, i, 0], pos[0, j, 0]],
                [pos[0, i, 1], pos[0, j, 1]],
                c=edge_colors[i] if edge_colors is not None else "gray",
                lw=1.5,
                alpha=0.6,
            )[0]

        edge_lines.append(line)

    def update(frame):
        # Update node positions
        if D == 2:
            scat.set_offsets(pos[frame])
        else:
            scat._offsets3d = (pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2])

        # Update node colors
        if node_colors is not None:
            if node_colors.ndim == 2:
                scat.set_color(node_colors[frame])
            else:
                scat.set_color(node_colors)

        # Update edge lines
        for line, (i, j) in zip(edge_lines, graph.edges):
            if D == 2:
                line.set_data(
                    [pos[frame, i, 0], pos[frame, j, 0]],
                    [pos[frame, i, 1], pos[frame, j, 1]],
                )
            else:
                line.set_data_3d(
                    [pos[frame, i, 0], pos[frame, j, 0]],
                    [pos[frame, i, 1], pos[frame, j, 1]],
                    [pos[frame, i, 2], pos[frame, j, 2]],
                )

        # Update orientations
        if ori is not None and D == 2:
            quivers.set_offsets(pos[frame])
            quivers.set_UVC(ori[frame, :, 0], ori[frame, :, 1])

        ax.set_title(f"{kwargs.get('title', 'Temporal Graph')}\nFrame {frame+1}/{T}")

        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.close()
    return ani


if __name__ == "__main__":
    T, N = 100, 10
    pos = np.random.randn(T, N, 2).cumsum(axis=0) * 0.1
    G = nx.path_graph(N)
    ani = animate_temporal_graph(pos, G, title="Random Walk on Path")
    ani.save("temporal_graph_animation.mp4", writer="ffmpeg", fps=30)
