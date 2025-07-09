import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


class ChiralMetamaterial:
    def __init__(
        self,
        graph,  # networkx.Graph with integer nodes 0…N-1
        pos,  # array (N, D) of initial positions
        D=2,  # embedding dimension
        L=1.0,  # nominal bar length
        k_e=1e4,  # bar‐spring stiffness
        r=0.2,  # winding radius
        k_r=10,  # rotational spring stiffness
        gamma=1,  # translational damping
        gamma_t=0.1,  # rotational damping
        dt=1e-4,  # timestep
        forcing_fn=None,  # NEW: function(t, state, F_int, T_int) → (F_ext, T_ext) adds forces
        vel_fn=None,  # NEW: function(t, state) → (vel_pos, vel_theta) adds velocities, fixes, etc.
    ):
        # topology & state
        self.graph = graph
        self.N = len(graph)
        self.D = D
        pos0 = jnp.array(pos).reshape(self.N, D)
        th0 = jnp.zeros((self.N,))
        self.state = jnp.concatenate([pos0, th0[:, None]], axis=1)
        self.vel_pos = jnp.zeros((self.N, D))
        self.vel_theta = jnp.zeros((self.N,))

        # physical parameters
        self.L, self.k_e = L, k_e
        self.r, self.k_r = r, k_r
        self.gamma, self.gamma_t = gamma, gamma_t
        self.dt = dt

        # time counter
        self.t = 0.0

        # store forcing function
        # signature: (t, state, F_int, T_int) -> (F_ext, T_ext)
        self.forcing_fn = forcing_fn
        # store velocity function
        # signature: (t, state, F_int, T_int, vel_pos, vel_theta) -> (vel_pos, vel_theta)
        self.vel_fn = vel_fn

        # precompute edge list for fast JIT
        edges = jnp.array(list(graph.edges()), dtype=jnp.int32)
        self.edges = edges

        # JIT‐compile energy and its gradient
        self._energy = jax.jit(self._energy_fn)
        self._grad = jax.jit(jax.grad(self._energy_fn))

    def _energy_fn(self, state):
        pos = state[:, : self.D]
        theta = state[:, self.D]

        # 1) bar‐spring with rotation‐shifted rest‐length
        i0, i1 = self.edges[:, 0], self.edges[:, 1]
        p0, p1 = pos[i0], pos[i1]
        d = jnp.linalg.norm(p0 - p1, axis=1)
        curr_L = self.L - self.r * (theta[i0] + theta[i1])
        E_edge = 0.5 * self.k_e * jnp.sum((d - curr_L) ** 2)

        # 2) rotational spring
        E_rot = 0.5 * self.k_r * jnp.sum(theta**2)

        # no “pin” terms here—those are now in your forcing_fn if you like

        return E_edge + E_rot

    def step(self):
        """One damped‐Euler step under internal forces + user forcing_fn."""
        state = self.state
        grad = self._grad(state)  # shape (N, D+1)

        # internal forces & torques
        F_int = -grad[:, : self.D]
        T_int = -grad[:, self.D]

        # user‐provided external forcing / constraints
        if self.forcing_fn is not None:
            F_ext, T_ext = self.forcing_fn(self.t, state, F_int, T_int)
        else:
            F_ext = jnp.zeros_like(self.vel_pos)
            T_ext = jnp.zeros_like(self.vel_theta)

        # add damping
        F = F_int + F_ext - self.gamma * self.vel_pos
        T = T_int + T_ext - self.gamma_t * self.vel_theta

        # update velocities
        self.vel_pos = self.vel_pos + self.dt * F
        self.vel_theta = self.vel_theta + self.dt * T

        if self.vel_fn is not None:
            # add user‐provided velocities
            vel_pos, vel_theta = self.vel_fn(
                self.t, state, F_int, T_int, self.vel_pos, self.vel_theta
            )
            self.vel_pos = vel_pos
            self.vel_theta = vel_theta

        # integrate positions & angles
        pos_new = state[:, : self.D] + self.dt * self.vel_pos
        theta_new = state[:, self.D] + self.dt * self.vel_theta

        # reassemble state
        self.state = jnp.concatenate([pos_new, theta_new[:, None]], axis=1)

        # advance time
        self.t += self.dt

        return self.state

    def simulate(self, steps):
        traj = []
        for _ in range(steps):
            traj.append(self.state)
            self.step()
        return traj


def make_vel_fn(bottom_idx, top_idx, v_top_z, t_release=20):
    """
    Returns a vel_fn(t, state, F_int, T_int, vel_pos, vel_theta)
    which:
      • zeroes out pos‑vel & ang‑vel on bottom_idx
      • forces z‑velocity = v_top_z on top_idx
    """
    # turn into jnp arrays for indexing
    bottom = jnp.array(bottom_idx, dtype=jnp.int32)
    top = jnp.array(top_idx, dtype=jnp.int32)

    def vel_fn(t, state, F_int, T_int, vel_pos, vel_theta):
        # 1) Pin bottom nodes: zero their velocities
        #    (you’ll also want to clamp their positions in step() separately)
        # vel_pos   = vel_pos.at[bottom].set(0.0)
        # vel_theta = vel_theta.at[bottom].set(0.0)
        if t < t_release:
            vel_pos = vel_pos.at[bottom, 2].set(0.0)
            # 2) Enforce constant z‑velocity on the top nodes
            #    (assumes self.D >= 3 so axis 2 is your “z”)
            vel_pos = vel_pos.at[top, 2].set(v_top_z)

        return vel_pos, vel_theta

    return vel_fn


def vel_fn_const_rot(bottom_idx, top_idx, rot_rate, t_release=20):
    """
    Returns a vel_fn(t, state, F_int, T_int, rot_rate, vel_theta)
    which:
      • Creates a constant rotation rate on all the nodes
    """
    # turn into jnp arrays for indexing
    bottom = jnp.array(bottom_idx, dtype=jnp.int32)
    top = jnp.array(top_idx, dtype=jnp.int32)

    def vel_fn(t, state, F_int, T_int, vel_pos, vel_theta):
        # 1) Pin bottom nodes: zero their velocities
        #    (you’ll also want to clamp their positions in step() separately)
        # vel_pos   = vel_pos.at[bottom].set(0.0)
        # vel_theta = vel_theta.at[bottom].set(0.0)

        if t < t_release:
            vel_theta = vel_theta.at[:].set(rot_rate)
            # vel_pos = vel_pos.at[bottom, 2].set(0.0)
            # # 2) Enforce constant z‑velocity on the top nodes
            # #    (assumes self.D >= 3 so axis 2 is your “z”)
            # vel_pos = vel_pos.at[top, 2].set(v_top_z)

        return vel_pos, vel_theta

    return vel_fn


def grid_graph_3d(m, n, k, boundary_type="line"):
    """
    Creates a grid graph with nodes at the top and the bottom given as lists.

    Parameters:
        m : max number of nodes in the x-direction
        n : max number of nodes in the y-direction
        k : max number of nodes in the z-direction
        boundary_type : 'line' for a line of nodes at the top and bottom,
                        'full' for a full boundary with edges.

    Returns:
        G : networkx.Graph representing the grid with boundaries
        pos: jax array of shape (N, 2) with node positions
        boundary_top_nodes : list of nodes at the top boundary
        boundary_bottom_nodes : list of nodes at the bottom boundary
    """
    G = nx.grid_graph((k, n, m))  # Create a grid graph

    G_int = nx.convert_node_labels_to_integers(
        G, ordering="default", label_attribute="orig_label"
    )

    # 2) Build reverse map: (x,y) → new integer
    orig2new = {data["orig_label"]: node for node, data in G_int.nodes(data=True)}
    # loop over all x, y, but choose z=0 and z=k-1
    top_nodes = [(i, j, k - 1) for i in range(m) for j in range(n)]
    bottom_nodes = [(i, j, 0) for i in range(m) for j in range(n)]

    top_int = [orig2new[t] for t in top_nodes]
    bottom_nodes_int = [orig2new[b] for b in bottom_nodes]
    # boundary_edges_int =  [(orig2new[u], orig2new[v]) for u,v in boundary_edges]
    pos_arr = jnp.array(
        [list(G_int.nodes[i]["orig_label"]) for i in G_int.nodes()], dtype=float
    )  # shape (N,2)

    return G_int, pos_arr, top_int, bottom_nodes_int


def animate_temporal_graph_test(
    pos,
    graph,
    ori=None,
    node_colors=None,  # either shape (N,) static scalars or (T,N) dynamic scalars or (T,N,4) RGBA
    cmap="viridis",  # name or Colormap instance, only used if node_colors is scalar
    vmin=None,
    vmax=None,  # color limits for scalar mode
    show_colorbar=False,
    edge_colors=None,
    interval=200,
    **kwargs,
):
    T, N, D = pos.shape
    if D not in [2, 3]:
        raise ValueError("Only 2D or 3D supported.")

    # to numpy
    pos = np.array(pos)
    if ori is not None:
        ori = np.array(ori)
    if isinstance(node_colors, np.ndarray):
        node_colors = np.array(node_colors)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d" if D == 3 else None)

    # compute limits (unchanged)...
    all_pos = pos.reshape(-1, D)
    mn, mx = all_pos.min(axis=0), all_pos.max(axis=0)
    pad = 0.1 * (mx - mn)
    if D == 3:
        ax.set_xlim3d(mn[0] - pad[0], mx[0] + pad[0])
        ax.set_ylim3d(mn[1] - pad[1], mx[1] + pad[1])
        ax.set_zlim3d(mn[2] - pad[2], mx[2] + pad[2])
    else:
        ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
        ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])

    ax.set_title(kwargs.get("title", "Temporal Graph"))

    # Determine “color mode”:
    color_mode = None
    if node_colors is None:
        color_mode = "none"
    elif node_colors.ndim == 3 and node_colors.shape[2] == 4:
        color_mode = "rgba"  # precomputed RGBA
    elif node_colors.ndim == 2:
        color_mode = "scalar"  # we map via cmap+norm
    elif node_colors.ndim == 1:
        color_mode = "static_scalar"
    else:
        raise ValueError(f"Unrecognized node_colors shape {node_colors.shape}")

    # Prepare Norm if scalar
    norm = None
    if color_mode in ("scalar", "static_scalar"):
        norm = Normalize(
            vmin=(vmin if vmin is not None else node_colors.min()),
            vmax=(vmax if vmax is not None else node_colors.max()),
        )

    # scatter initial
    if color_mode == "rgba":
        init_c = node_colors[0]
        scat = ax.scatter(
            *([pos[0, :, i] for i in range(D)]), c=init_c, s=100, zorder=2
        )
    elif color_mode == "scalar":
        init_vals = node_colors[0]
        scat = ax.scatter(
            *([pos[0, :, i] for i in range(D)]),
            c=init_vals,
            cmap=cmap,
            norm=norm,
            s=100,
            zorder=2,
        )
    elif color_mode == "static_scalar":
        scat = ax.scatter(
            *([pos[0, :, i] for i in range(D)]),
            c=node_colors,
            cmap=cmap,
            norm=norm,
            s=100,
            zorder=2,
        )
    else:
        scat = ax.scatter(
            *([pos[0, :, i] for i in range(D)]), color="C0", s=100, zorder=2
        )

    # add colorbar if wanted
    if show_colorbar and color_mode in ("scalar", "static_scalar"):
        fig.colorbar(scat, ax=ax)

    # draw edges once
    edge_lines = []
    for i, j in graph.edges:
        if D == 2:
            line = ax.plot(
                [pos[0, i, 0], pos[0, j, 0]],
                [pos[0, i, 1], pos[0, j, 1]],
                c=edge_colors[i] if edge_colors is not None else "gray",
                lw=1.5,
                alpha=0.6,
            )[0]
        else:
            line = ax.plot(
                [pos[0, i, 0], pos[0, j, 0]],
                [pos[0, i, 1], pos[0, j, 1]],
                [pos[0, i, 2], pos[0, j, 2]],
                c=edge_colors[i] if edge_colors is not None else "gray",
                lw=1.5,
                alpha=0.6,
            )[0]
        edge_lines.append(line)

    # (ori/quiver logic elided for brevity) …

    def update(frame):
        # positions
        if D == 2:
            scat.set_offsets(pos[frame, :, :2])
        else:
            scat._offsets3d = (pos[frame, :, 0], pos[frame, :, 1], pos[frame, :, 2])

        # update colors
        if color_mode == "rgba":
            scat.set_facecolors(node_colors[frame])
        elif color_mode == "scalar":
            scat.set_array(node_colors[frame])
        # static_scalar stays unchanged

        # edges
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

        ax.set_title(f"{kwargs.get('title','')}\nFrame {frame+1}/{T}")
        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.close()
    return ani


if __name__ == "__main__":
    m = 8
    n = 8
    k = 8
    # after you build your grid:
    G, pos0, top_idx, bottom_idx = grid_graph_3d(m, n, k)

    # make a vel_fn that pins the bottom and drives the top downward at 0.05 units/s
    my_vel_fn = make_vel_fn(bottom_idx, top_idx, v_top_z=-0.25)

    # const_rotation = vel_fn_const_rot(bottom_idx, top_idx, rot_rate = 0.15)

    mat = ChiralMetamaterial(
        G,
        pos0,
        D=3,
        dt=1e-3,
        k_r=1e6,
        k_e=1e2,
        forcing_fn=None,  # or whatever extra forces you like
        vel_fn=my_vel_fn,  # const_rotation
    )

    # simulate for t_f = 10 s
    t_f = 10.0
    steps = int(t_f / mat.dt)
    traj = mat.simulate(steps)
    # sample every 10 steps
    traj = traj[::100]
    traj = jnp.array(traj)
    pos = traj[:, :, :3]
    pos = np.array(pos)
    angle = np.array(traj[:, :, 3])
    # # map angle to color
    # cmap = plt.get_cmap('hsv')
    # norm = plt.Normalize(vmin=np.min(angle), vmax=np.max(angle))
    # angle = cmap(norm(angle))

    ani = animate_temporal_graph_test(
        pos,
        G,
        title="3D Non-Chiral Metamaterial",
        node_colors=angle,  # pass scalars shape (T,N)
        cmap="plasma",  # choose HSV mapping
        show_colorbar=True,
        interval=100,
    )
    ani.save("chiral_3d_compression_no_chiral.mp4", writer="ffmpeg", fps=10)
    print(traj.shape)

    # not to self need to create plotting code to set the color of the ndoes to refelect the chirality
