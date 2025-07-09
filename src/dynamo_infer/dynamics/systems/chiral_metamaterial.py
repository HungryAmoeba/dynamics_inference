import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional, Callable

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from ..base import DynamicalSystem
from ...config.schemas import DynamicsConfig


class ChiralMetamaterial(DynamicalSystem):
    """
    Chiral metamaterial system with graph-based topology and dynamics.
    
    The system consists of nodes with:
    - Spatial positions (x, y, z)
    - Rotation angles θ
    - Position velocities
    - Angular velocities
    
    The dynamics include:
    - Elastic forces from deformed edges
    - Rotational spring forces
    - Damping
    - Optional external forcing and velocity constraints
    """
    
    def __init__(self):
        super().__init__()
        # Physical parameters
        self.L = None  # nominal bar length
        self.k_e = None  # bar-spring stiffness
        self.r = None  # winding radius
        self.k_r = None  # rotational spring stiffness
        self.gamma = None  # translational damping
        self.gamma_t = None  # rotational damping
        self.dt = None  # timestep
        
        # Graph structure
        self.graph = None
        self.edges = None
        
        # Functions for external forces/constraints
        self.forcing_fn = None
        self.vel_fn = None
        
        # JAX compiled functions
        self._energy = None
        self._grad = None
        
        # Current time
        self.t = 0.0

    def initialize(self, config: DynamicsConfig) -> None:
        """Initialize the chiral metamaterial system."""
        super().initialize(config)
        
        # Get parameters from config
        params = config.parameters
        self.L = params.get("L", 1.0)
        self.k_e = params.get("k_e", 1e4)
        self.r = params.get("r", 0.2)
        self.k_r = params.get("k_r", 10)
        self.gamma = params.get("gamma", 1.0)
        self.gamma_t = params.get("gamma_t", 0.1)
        self.dt = params.get("dt", 1e-4)
        
        # Grid parameters for creating lattice
        grid_type = params.get("grid_type", "3d")
        grid_size = params.get("grid_size", [8, 8, 8])
        
        if isinstance(grid_size, int):
            if grid_type == "2d":
                grid_size = [grid_size, grid_size]
            else:
                grid_size = [grid_size, grid_size, grid_size]
        
        # Initialize graph and positions
        if grid_type == "3d":
            m, n, k = grid_size
            self.graph, pos0, top_idx, bottom_idx = grid_graph_3d(m, n, k)
        else:
            raise ValueError(f"Grid type {grid_type} not yet supported")
        
        # Update n_particles and dimension from the actual graph
        self.n_particles = len(self.graph)
        self.dimension = pos0.shape[1]
        
        # Store forcing and velocity functions if provided
        forcing_params = params.get("forcing", {})
        if forcing_params.get("type") == "velocity_constraint":
            v_top_z = forcing_params.get("v_top_z", -0.25)
            t_release = forcing_params.get("t_release", 20.0)
            self.vel_fn = make_vel_fn(bottom_idx, top_idx, v_top_z, t_release)
        elif forcing_params.get("type") == "constant_rotation":
            rot_rate = forcing_params.get("rot_rate", 0.15)
            t_release = forcing_params.get("t_release", 20.0)
            self.vel_fn = vel_fn_const_rot(bottom_idx, top_idx, rot_rate, t_release)
            
        self.forcing_fn = params.get("forcing_fn", None)
        
        # Precompute edge list for fast JIT
        self.edges = jnp.array(list(self.graph.edges()), dtype=jnp.int32)
        
        # JIT-compile energy and its gradient
        self._energy = jax.jit(self._energy_fn)
        self._grad = jax.jit(jax.grad(self._energy_fn))
        
        # Initialize state: [positions, angles, position_velocities, angular_velocities]
        initial_conditions = config.initial_conditions
        pos_noise = initial_conditions.get("position_noise", 0.01)
        angle_noise = initial_conditions.get("angle_noise", 0.01)
        vel_noise = initial_conditions.get("velocity_noise", 0.01)
        
        # Use provided seed for reproducibility
        seed = params.get("seed", 42)
        rng = jax.random.PRNGKey(seed)
        
        # Initial positions with small random perturbations
        rng, rng_pos = jax.random.split(rng)
        positions = pos0 + pos_noise * jax.random.normal(rng_pos, pos0.shape)
        
        # Initial angles
        rng, rng_ang = jax.random.split(rng)
        angles = angle_noise * jax.random.normal(rng_ang, (self.n_particles,))
        
        # Initial velocities
        rng, rng_vel = jax.random.split(rng)
        position_velocities = vel_noise * jax.random.normal(rng_vel, (self.n_particles, self.dimension))
        
        rng, rng_avel = jax.random.split(rng)
        angular_velocities = vel_noise * jax.random.normal(rng_avel, (self.n_particles,))
        
        # Create state vector
        self.state = self.ravel_state(positions, angles, position_velocities, angular_velocities)
        self.initialized = True
        
    def _energy_fn(self, positions: jnp.ndarray, theta: jnp.ndarray) -> float:
        """
        Compute the total energy of the system.
        
        Args:
            positions: (N, D) array of particle positions
            theta: (N,) array of rotation angles
            
        Returns:
            Total energy
        """
        if self.edges is None or self.L is None or self.r is None or self.k_e is None or self.k_r is None:
            raise ValueError("System not properly initialized")
            
        # 1) bar-spring with rotation-shifted rest-length
        i0, i1 = self.edges[:, 0], self.edges[:, 1]
        p0, p1 = positions[i0], positions[i1]
        d = jnp.linalg.norm(p0 - p1, axis=1)
        curr_L = self.L - self.r * (theta[i0] + theta[i1])
        E_edge = 0.5 * self.k_e * jnp.sum((d - curr_L) ** 2)

        # 2) rotational spring
        E_rot = 0.5 * self.k_r * jnp.sum(theta**2)

        return E_edge + E_rot

    def compute_derivatives(self, t: float, state: jnp.ndarray, args: Any = None) -> jnp.ndarray:
        """
        Compute the time derivatives using damped dynamics.
        
        Args:
            t: Current time
            state: Current state vector
            args: Additional arguments (unused)
            
        Returns:
            Time derivatives of the state
        """
        positions, angles, pos_vel, ang_vel = self._unwrap_state(state)
        
        # Compute energy gradient
        grad_fn = jax.grad(self._energy_fn, argnums=(0, 1))
        grad_pos, grad_theta = grad_fn(positions, angles)
        
        # Internal forces & torques
        F_int = -grad_pos
        T_int = -grad_theta
        
        # External forcing
        if self.forcing_fn is not None:
            # Reconstruct state in original format for compatibility
            state_orig = jnp.concatenate([positions, angles[:, None]], axis=1)
            F_ext, T_ext = self.forcing_fn(t, state_orig, F_int, T_int)
        else:
            F_ext = jnp.zeros_like(pos_vel)
            T_ext = jnp.zeros_like(ang_vel)
            
        # Total forces with damping
        F = F_int + F_ext - self.gamma * pos_vel
        T = T_int + T_ext - self.gamma_t * ang_vel
        
        # Velocity derivatives (accelerations)
        pos_vel_dot = F
        ang_vel_dot = T
        
        # Apply velocity constraints if present
        if self.vel_fn is not None:
            # Reconstruct state in original format for compatibility
            state_orig = jnp.concatenate([positions, angles[:, None]], axis=1)
            pos_vel_new, ang_vel_new = self.vel_fn(t, state_orig, F_int, T_int, pos_vel, ang_vel)
            
            # Override velocities if constraints are active
            pos_vel = pos_vel_new
            ang_vel = ang_vel_new
            
            # Set accelerations to zero for constrained degrees of freedom
            # This is a simplified approach; more sophisticated constraint handling could be implemented
        
        # Position derivatives (velocities)
        pos_dot = pos_vel
        ang_dot = ang_vel
        
        return self.ravel_state(pos_dot, ang_dot, pos_vel_dot, ang_vel_dot)

    def return_state(self) -> jnp.ndarray:
        """Return the current state of the system."""
        return self.state

    def unwrap_state(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Unwrap the state into a dictionary of named components.
        Works across the entire trajectory.
        
        Args:
            state: State array of shape (state_dim,) for single state or (T, state_dim) for trajectory
            
        Returns:
            Dictionary with keys:
            - 'positions': (N, D) or (T, N, D) array for positions
            - 'angles': (N,) or (T, N) array for rotation angles
            - 'position_velocities': (N, D) or (T, N, D) array for position velocities
            - 'angular_velocities': (N,) or (T, N) array for angular velocities
            - 'orientations': (N, D) or (T, N, D) array for orientations (computed from angles)
        """
        # Handle both single states and trajectories
        if len(state.shape) == 1:
            # Single state: (state_dim,)
            positions, angles, pos_vel, ang_vel = self._unwrap_state(state)
            
            # Compute orientations from angles (2D rotation for now)
            if self.dimension >= 2:
                cos_theta = jnp.cos(angles)
                sin_theta = jnp.sin(angles)
                if self.dimension == 2:
                    orientations = jnp.stack([cos_theta, sin_theta], axis=1)
                else:
                    # For 3D, assume rotation around z-axis
                    orientations = jnp.stack([cos_theta, sin_theta, jnp.zeros_like(angles)], axis=1)
            else:
                orientations = angles[:, None]
        else:
            # Trajectory: (T, state_dim)
            T = state.shape[0]
            positions_list, angles_list, pos_vel_list, ang_vel_list = [], [], [], []
            orientations_list = []
            
            for i in range(T):
                pos, ang, pv, av = self._unwrap_state(state[i])
                positions_list.append(pos)
                angles_list.append(ang)
                pos_vel_list.append(pv)
                ang_vel_list.append(av)
                
                # Compute orientations
                if self.dimension >= 2:
                    cos_theta = jnp.cos(ang)
                    sin_theta = jnp.sin(ang)
                    if self.dimension == 2:
                        orient = jnp.stack([cos_theta, sin_theta], axis=1)
                    else:
                        orient = jnp.stack([cos_theta, sin_theta, jnp.zeros_like(ang)], axis=1)
                else:
                    orient = ang[:, None]
                orientations_list.append(orient)
            
            positions = jnp.array(positions_list)
            angles = jnp.array(angles_list)
            pos_vel = jnp.array(pos_vel_list)
            ang_vel = jnp.array(ang_vel_list)
            orientations = jnp.array(orientations_list)

        return {
            "positions": positions,
            "angles": angles,
            "position_velocities": pos_vel,
            "angular_velocities": ang_vel,
            "orientations": orientations,
        }

    def get_expected_state_shape(self) -> Tuple[int, ...]:
        """Get the expected shape of the state vector."""
        # State includes: positions (N*D) + angles (N) + position_velocities (N*D) + angular_velocities (N)
        return (self.n_particles * self.dimension + self.n_particles + 
                self.n_particles * self.dimension + self.n_particles,)

    def ravel_state(self, positions: jnp.ndarray, angles: jnp.ndarray, 
                   pos_vel: jnp.ndarray, ang_vel: jnp.ndarray) -> jnp.ndarray:
        """Ravel state components into a single state vector."""
        return jnp.concatenate([
            positions.ravel(),  # (N*D,)
            angles.ravel(),     # (N,)
            pos_vel.ravel(),    # (N*D,)
            ang_vel.ravel(),    # (N,)
        ])

    def _unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Unwrap the state vector into its components.
        
        Args:
            state: Flat state vector
            
        Returns:
            Tuple of (positions, angles, position_velocities, angular_velocities)
        """
        N, D = self.n_particles, self.dimension
        
        # Split state vector
        pos_size = N * D
        positions = state[:pos_size].reshape(N, D)
        angles = state[pos_size:pos_size + N]
        pos_vel = state[pos_size + N:pos_size + N + pos_size].reshape(N, D)
        ang_vel = state[pos_size + N + pos_size:]
        
        return positions, angles, pos_vel, ang_vel

    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure."""
        if self.graph is None:
            raise ValueError("Graph not initialized")
        return {
            "n_nodes": len(self.graph),
            "n_edges": len(self.graph.edges()),
            "graph": self.graph,
            "edges": self.edges,
        }

    def to_networkx_graph(self):
        """Return the underlying NetworkX graph."""
        return self.graph

    def step_simulation(self):
        """
        Perform one simulation step (for compatibility with original interface).
        This integrates the dynamics by one timestep.
        """
        if self.state is None or self.dt is None:
            raise ValueError("System not properly initialized")
            
        # Compute derivatives
        derivatives = self.compute_derivatives(self.t, self.state)
        
        # Simple Euler integration
        self.state = self.state + self.dt * derivatives
        self.t += self.dt
        
        return self.state

    def simulate(self, steps: int):
        """
        Run simulation for a given number of steps.
        
        Args:
            steps: Number of simulation steps
            
        Returns:
            List of states at each timestep
        """
        if self.state is None:
            raise ValueError("System not properly initialized")
            
        trajectory = []
        for _ in range(steps):
            trajectory.append(self.state.copy())
            self.step_simulation()
        return trajectory


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
    from ...config.schemas import DynamicsConfig
    
    m = 8
    n = 8
    k = 8

    mat = ChiralMetamaterial()
    config = DynamicsConfig(
        type="chiral_metamaterial",
        n_particles=0,  # Will be set by the system
        dimension=3,
        parameters={
            "L": 1.0, "k_e": 1e4, "r": 0.2, "k_r": 1e6, 
            "gamma": 1.0, "gamma_t": 0.1, "dt": 1e-3,
            "grid_type": "3d", "grid_size": [m, n, k], 
            "forcing": {"type": "velocity_constraint", "v_top_z": -0.25, "t_release": 20.0},
            "seed": 42
        },
        initial_conditions={
            "position_noise": 0.01, 
            "angle_noise": 0.01, 
            "velocity_noise": 0.01
        }
    )
    mat.initialize(config)

    # simulate for t_f = 10 s
    t_f = 10.0
    steps = int(t_f / mat.dt)
    traj = mat.simulate(steps)
    
    # sample every 10 steps
    traj_sampled = traj[::100]
    traj_array = jnp.array(traj_sampled)
    
    # Extract positions and angles using the unwrap_state method
    unwrapped = mat.unwrap_state(traj_array)
    pos = np.array(unwrapped["positions"])
    angles = np.array(unwrapped["angles"])

    ani = animate_temporal_graph_test(
        pos,
        mat.graph,
        title="3D Chiral Metamaterial",
        node_colors=angles,  # pass scalars shape (T,N)
        cmap="plasma",  # choose HSV mapping
        show_colorbar=True,
        interval=100,
    )
    ani.save("chiral_3d_metamaterial.mp4", writer="ffmpeg", fps=10)
    print(f"Trajectory shape: {traj_array.shape}")
    print(f"Positions shape: {pos.shape}")
    print(f"Angles shape: {angles.shape}")
