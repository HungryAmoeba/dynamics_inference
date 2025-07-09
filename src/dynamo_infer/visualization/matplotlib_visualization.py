import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
from .temporal_graph_matplotlib import animate_temporal_graph


def to_numpy(arr):
    if arr is not None and not isinstance(arr, np.ndarray):
        try:
            return np.array(arr)
        except Exception:
            pass
    return arr


def rescale_node_sizes(sizes, min_size=50, max_size=500, shift=True):
    sizes = np.asarray(sizes)
    if sizes.ndim == 1:
        sizes = sizes[None, :]  # (1, N) for static
    min_val = np.min(sizes)
    max_val = np.max(sizes)
    if shift:
        sizes = sizes - min_val  # shift to start at 0
        max_val = np.max(sizes)
        min_val = 0
    if max_val > min_val:
        scaled = min_size + (sizes - min_val) * (max_size - min_size) / (
            max_val - min_val
        )
    else:
        scaled = np.full_like(sizes, (min_size + max_size) / 2)
    return scaled.squeeze()


class MatplotlibVisualizer:
    """
    Visualizer for particle dynamics using matplotlib temporal graph animations.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def visualize(
        self,
        pos,
        ori=None,
        save_path=None,
        graph=None,
        node_colors=None,
        edge_colors=None,
        node_sizes=None,
        **kwargs,
    ):
        """
        Visualize particle motion using matplotlib temporal graph animation.
        Args:
            pos: Array of shape (T, N, D) where D=2 or 3
            ori: Optional array of shape (T, N, D) for orientations
            save_path: Path to save the animation (mp4)
            graph: Optional networkx.Graph object. If None, use no edges.
            node_colors: Optional array for node colors
            edge_colors: Optional array for edge colors
            node_sizes: Optional array for node sizes (T, N) or (N,)
            **kwargs: Additional arguments (e.g., interval, title)
        Returns:
            anim: The matplotlib.animation.FuncAnimation object
        """
        # Robustly convert all arrays to numpy
        pos = to_numpy(pos)
        ori = to_numpy(ori)
        node_colors = to_numpy(node_colors)
        edge_colors = to_numpy(edge_colors)
        node_sizes = to_numpy(node_sizes)
        interval = getattr(self.config, "interval", 50) if self.config else 50
        title = getattr(self.config, "title", None) if self.config else None
        if "interval" in kwargs:
            interval = kwargs.pop("interval")
        if "title" in kwargs:
            title = kwargs.pop("title")
        if graph is None:
            # Create a graph with N nodes and no edges
            N = pos.shape[1]
            graph = nx.empty_graph(N)
        # Shape check
        if pos.ndim != 3:
            raise ValueError(f"pos must be (T, N, D), got shape {pos.shape}")
        if ori is not None and (ori.ndim != 3 and ori.ndim != 2):
            raise ValueError(f"ori must be (T, N, D) or (T, N), got shape {ori.shape}")
        # Rescale node sizes if provided
        if node_sizes is not None:
            node_sizes = rescale_node_sizes(node_sizes)
        anim = animate_temporal_graph(
            pos,
            graph,
            ori=ori,
            node_colors=node_colors,
            edge_colors=edge_colors,
            node_sizes=node_sizes,
            interval=interval,
            title=title,
            **kwargs,
        )
        if save_path is not None:
            fps = int(1000 / interval)
            anim.save(
                save_path, writer="ffmpeg", fps=fps, extra_args=["-vcodec", "libx264"]
            )
        return anim


def animate_particle_motion(pos, ori=None, interval=50, title=None, **kwargs):
    """
    Animate particle positions with optional orientation arrows.
    Args:
        pos: Array of shape (T, N, D) where D=2 or 3
        ori: Optional array of shape (T, N, D) for orientations
        interval: Delay between frames in milliseconds
        title: Title for the animation
    Returns:
        anim: The matplotlib.animation.FuncAnimation object
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(pos, "device_buffer"):
        pos = np.array(pos)
    if ori is not None and hasattr(ori, "device_buffer"):
        ori = np.array(ori)

    T, N, D = pos.shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D data supported")

    fig = plt.figure(figsize=(8, 6))
    if D == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=20, azim=45)
    else:
        ax = fig.add_subplot(111)

    all_pos = pos.reshape(-1, D)
    min_vals = all_pos.min(axis=0)
    max_vals = all_pos.max(axis=0)
    padding = 0.1 * (max_vals - min_vals)

    if D == 3:
        ax.set_xlim3d(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim3d(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim3d(min_vals[2] - padding[2], max_vals[2] + padding[2])
    else:
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])

    if D == 2:
        scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], c="blue", s=50, alpha=0.8)
    else:
        scat = ax.scatter(
            pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], c="blue", s=50, alpha=0.8
        )

    quivers = None
    arrow_scale = 0.1 * (max_vals - min_vals).mean() if ori is not None else None

    if ori is not None:
        if D == 3:
            quivers = ax.quiver(
                pos[0, :, 0],
                pos[0, :, 1],
                pos[0, :, 2],
                ori[0, :, 0] * arrow_scale,
                ori[0, :, 1] * arrow_scale,
                ori[0, :, 2] * arrow_scale,
                length=arrow_scale,
                normalize=True,
                color="red",
            )
        else:
            quivers = ax.quiver(
                pos[0, :, 0],
                pos[0, :, 1],
                ori[0, :, 0] * arrow_scale,
                ori[0, :, 1] * arrow_scale,
                color="red",
                scale_units="xy",
                scale=1 / arrow_scale,
            )

    def update(frame):
        if D == 2:
            scat.set_offsets(pos[frame])
        else:
            scat._offsets3d = (pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2])
        if ori is not None and quivers is not None:
            if D == 3:
                for artist in ax.collections[:]:
                    if artist not in [scat]:
                        artist.remove()
                ax.quiver(
                    pos[frame, :, 0],
                    pos[frame, :, 1],
                    pos[frame, :, 2],
                    ori[frame, :, 0] * arrow_scale,
                    ori[frame, :, 1] * arrow_scale,
                    ori[frame, :, 2] * arrow_scale,
                    length=arrow_scale,
                    normalize=True,
                    color="red",
                )
            else:
                quivers.set_offsets(pos[frame])
                quivers.set_UVC(
                    ori[frame, :, 0] * arrow_scale, ori[frame, :, 1] * arrow_scale
                )
        ax.set_title(f"{title or 'Particle Motion'}\nFrame {frame}/{T}")
        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=True)
    plt.close()
    return ani


def generate_test_data(dim=2, num_particles=10, num_frames=100):
    """Generate test data with predictable patterns"""
    t = np.linspace(0, 4 * np.pi, num_frames)

    if dim == 2:
        # Circular motion with tangent orientations
        pos = np.zeros((num_frames, num_particles, 2))
        ori = np.zeros_like(pos)

        for i in range(num_particles):
            radius = 1 + 0.2 * i
            phase = 2 * np.pi * i / num_particles

            # Positions: rotating circles
            pos[:, i, 0] = radius * np.cos(t + phase)
            pos[:, i, 1] = radius * np.sin(t + phase)

            # Orientations: tangent to the circle
            ori[:, i, 0] = -np.sin(t + phase)  # x-component of tangent
            ori[:, i, 1] = np.cos(t + phase)  # y-component of tangent

    elif dim == 3:
        # Helical motion with spiral orientations
        pos = np.zeros((num_frames, num_particles, 3))
        ori = np.zeros_like(pos)

        for i in range(num_particles):
            radius = 1 + 0.2 * i
            phase = 2 * np.pi * i / num_particles
            z_speed = 0.5

            # Positions: rotating helices
            pos[:, i, 0] = radius * np.cos(t + phase)
            pos[:, i, 1] = radius * np.sin(t + phase)
            pos[:, i, 2] = z_speed * t

            # Orientations: tangent to the helix
            ori[:, i, 0] = -np.sin(t + phase)  # x-component
            ori[:, i, 1] = np.cos(t + phase)  # y-component
            ori[:, i, 2] = z_speed  # z-component

            # Normalize orientations
            norm = np.linalg.norm(ori[:, i, :], axis=1, keepdims=True)
            ori[:, i, :] /= norm

    return pos, ori


if __name__ == "__main__":
    # Generate and visualize test data
    pos_2d, ori_2d = generate_test_data(dim=2)
    pos_3d, ori_3d = generate_test_data(dim=3)

    # Create animations
    anim_2d = animate_particle_motion(pos_2d, ori_2d, title="2D Circular Motion Test")
    anim = animate_particle_motion(pos_3d, ori_3d, title="3D Helical Motion Test")

    anim.save("3d_test.mp4", writer="ffmpeg", fps=30)
    anim_2d.save("2d_test.mp4", writer="ffmpeg", fps=30)
    # Display animations
    # HTML(anim_2d.to_html5_video())
    # HTML(anim_3d.to_html5_video())

# To save to files:
# anim_2d.save("2d_test.mp4", writer="ffmpeg", fps=30)
# anim_3d.save("3d_test.mp4", writer="ffmpeg", fps=30)
