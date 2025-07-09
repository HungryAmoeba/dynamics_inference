"""Lattice Hamiltonian system with overdamped dynamics."""

import jax.numpy as jnp
import jax.random as jrandom
import jax
from typing import Tuple, Any, Dict, List, Optional, Callable
from ..base import DynamicalSystem
from ...config.schemas import DynamicsConfig


class LatticeHamiltonianSystem(DynamicalSystem):
    """
    Lattice Hamiltonian system with overdamped dynamics.

    The system consists of lattice sites with:
    - Spatial locations x_i
    - Volumes A_i
    - Phases θ_i
    - Displacement field u_i
    - Volume field v_i (affects volumes as A_i + v_i)

    The Hamiltonian is:
    H = (k/2) * sum_{n ∈ N(i)} |u_j - u_i|² + (c/2) * sum_i v_i² +
        (λ/2) * sum_i (v_i - b(∇·u)_i)²

    Overdamped dynamics:
    γ_u * u̇_i = -∂H/∂u_i
    γ_v * v̇_i = -∂H/∂v_i + F_drive_i(t)

    where F_drive_i(t) is an optional external driving force.
    """

    def __init__(self):
        super().__init__()
        # Hamiltonian parameters
        self.k = None  # Elastic constant
        self.c = None  # Volume stiffness
        self.lambda_param = None  # Coupling parameter
        self.b = None  # Coupling coefficient
        self.kappa = None  # NEW: penalty for norm of u_i

        # Damping coefficients
        self.gamma_u = None
        self.gamma_v = None

        # Lattice structure
        self.lattice_positions = None  # Original lattice positions x_i
        self.volumes = None  # Base volumes A_i
        self.phases = None  # Phases θ_i
        self.neighbors = None  # Neighbor lists for each site

        # Grid parameters
        self.grid_shape = None  # (nx, ny) or (nx, ny, nz)
        self.lattice_spacing = None

        # Driving force parameters (volume)
        self.driving_forces = None  # List of driving force functions for each site
        self.driving_amplitude = None  # Amplitude of driving force
        self.driving_frequency = None  # Frequency of driving force
        self.driven_sites = None  # List of site indices to apply driving force
        self.driving_phases = None  # List of phases for each driven site (volume)

        # Displacement driving parameters (NEW)
        self.displacement_driving_amplitude = None
        self.displacement_driving_frequency = None
        self.displacement_driven_sites = None
        self.displacement_driving_phases = None
        self.displacement_driving_directions = None

        self.rng = None

    def initialize(self, config: DynamicsConfig) -> None:
        """
        Initialize the lattice Hamiltonian system.

        Args:
            config: DynamicsConfig object specifying system parameters
        """
        super().initialize(config)

        # Get parameters from config
        params = config.parameters
        self.k = params.get("k", 1.0)  # Elastic constant
        self.c = params.get("c", 1.0)  # Volume stiffness
        self.lambda_param = params.get("lambda", 0.5)  # Coupling parameter
        self.b = params.get("b", 1.0)  # Coupling coefficient
        self.kappa = params.get("kappa", 0.0)  # NEW: penalty for norm of u_i
        self.gamma_u = params.get("gamma_u", 1.0)  # Displacement damping
        self.gamma_v = params.get("gamma_v", 1.0)  # Volume damping
        self.lattice_spacing = params.get("lattice_spacing", 1.0)

        # Driving force parameters (volume)
        self.driving_amplitude = params.get("driving_amplitude", 0.0)
        self.driving_frequency = params.get("driving_frequency", 1.0)
        self.driven_sites = params.get(
            "driven_sites", []
        )  # List of site indices to drive
        self.driving_phases = params.get(
            "driving_phases", [0.0] * len(self.driven_sites)
        )
        if len(self.driving_phases) < len(self.driven_sites):
            # Pad with zeros if not enough phases specified
            self.driving_phases = self.driving_phases + [0.0] * (
                len(self.driven_sites) - len(self.driving_phases)
            )

        # Displacement driving parameters (NEW)
        self.displacement_driving_amplitude = params.get(
            "displacement_driving_amplitude", 0.0
        )
        self.displacement_driving_frequency = params.get(
            "displacement_driving_frequency", 1.0
        )
        self.displacement_driven_sites = params.get("displacement_driven_sites", [])
        self.displacement_driving_phases = params.get(
            "displacement_driving_phases", [0.0] * len(self.displacement_driven_sites)
        )
        if len(self.displacement_driving_phases) < len(self.displacement_driven_sites):
            self.displacement_driving_phases = self.displacement_driving_phases + [
                0.0
            ] * (
                len(self.displacement_driven_sites)
                - len(self.displacement_driving_phases)
            )
        # Directions: list of vectors (not normalized)
        default_direction = [1.0] + [0.0] * (params.get("dimension", 2) - 1)
        if "displacement_driving_directions" in params:
            dirs = params["displacement_driving_directions"]
            # Pad with default if not enough
            if len(dirs) < len(self.displacement_driven_sites):
                dirs = dirs + [default_direction] * (
                    len(self.displacement_driven_sites) - len(dirs)
                )
        else:
            dirs = [default_direction] * len(self.displacement_driven_sites)
        # Normalize directions
        import numpy as np

        normed_dirs = []
        for d in dirs:
            arr = np.array(d, dtype=float)
            norm = np.linalg.norm(arr)
            if norm == 0:
                arr = np.array(default_direction, dtype=float)
                norm = np.linalg.norm(arr)
            arr = arr / norm
            normed_dirs.append(arr.tolist())
        self.displacement_driving_directions = normed_dirs

        # Grid parameters
        grid_type = params.get("grid_type", "2d")  # "2d" or "3d"
        grid_size = params.get("grid_size", 5)  # Size of grid (e.g., 5x5 or 5x5x5)

        if grid_type == "2d":
            self.grid_shape = (grid_size, grid_size)
            self.dimension = 2
        elif grid_type == "3d":
            self.grid_shape = (grid_size, grid_size, grid_size)
            self.dimension = 3
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        # Initialize lattice
        self._initialize_lattice()

        # Get initial conditions
        initial_conditions = config.initial_conditions
        u_range = initial_conditions.get("displacement_range", [-0.1, 0.1])
        v_range = initial_conditions.get("volume_range", [-0.1, 0.1])

        # Use provided RNG key or a default one
        seed = params.get("seed", 42)
        self.rng = jrandom.PRNGKey(seed)

        # Initialize displacement and volume fields
        self.rng, rng_u = jrandom.split(self.rng)
        self.rng, rng_v = jrandom.split(self.rng)

        displacements = jrandom.uniform(
            rng_u,
            (self.n_particles, self.dimension),
            minval=u_range[0],
            maxval=u_range[1],
        )

        volume_fields = jrandom.uniform(
            rng_v,
            (self.n_particles,),
            minval=v_range[0],
            maxval=v_range[1],
        )

        # Create state: [displacements, volume_fields]
        self.state = self.ravel_state(displacements, volume_fields)
        self.initialized = True

    def _initialize_lattice(self):
        """Initialize the lattice structure and neighbor lists."""
        if self.dimension == 2:
            nx, ny = self.grid_shape
            self.n_particles = nx * ny

            # Create 2D grid positions
            x_coords = jnp.arange(nx) * self.lattice_spacing
            y_coords = jnp.arange(ny) * self.lattice_spacing
            X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij")
            self.lattice_positions = jnp.stack([X.flatten(), Y.flatten()], axis=1)

            # Create neighbor lists for 2D grid
            self.neighbors = self._create_2d_neighbors(nx, ny)

        elif self.dimension == 3:
            nx, ny, nz = self.grid_shape
            self.n_particles = nx * ny * nz

            # Create 3D grid positions
            x_coords = jnp.arange(nx) * self.lattice_spacing
            y_coords = jnp.arange(ny) * self.lattice_spacing
            z_coords = jnp.arange(nz) * self.lattice_spacing
            X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
            self.lattice_positions = jnp.stack(
                [X.flatten(), Y.flatten(), Z.flatten()], axis=1
            )

            # Create neighbor lists for 3D grid
            self.neighbors = self._create_3d_neighbors(nx, ny, nz)

        # Initialize base volumes and phases
        self.volumes = jnp.ones(self.n_particles)  # Unit volumes
        self.phases = jnp.zeros(self.n_particles)  # Zero phases

    def _create_2d_neighbors(self, nx: int, ny: int) -> List[List[int]]:
        """Create neighbor lists for 2D grid."""
        neighbors = []

        for i in range(nx):
            for j in range(ny):
                site_idx = i * ny + j
                site_neighbors = []

                # Check all 4 nearest neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        neighbor_idx = ni * ny + nj
                        site_neighbors.append(neighbor_idx)

                neighbors.append(site_neighbors)

        return neighbors

    def _create_3d_neighbors(self, nx: int, ny: int, nz: int) -> List[List[int]]:
        """Create neighbor lists for 3D grid."""
        neighbors = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    site_idx = i * (ny * nz) + j * nz + k
                    site_neighbors = []

                    # Check all 6 nearest neighbors
                    for di, dj, dk in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                            neighbor_idx = ni * (ny * nz) + nj * nz + nk
                            site_neighbors.append(neighbor_idx)

                    neighbors.append(site_neighbors)

        return neighbors

    def compute_hamiltonian(
        self, displacements: jnp.ndarray, volume_fields: jnp.ndarray
    ) -> float:
        """
        Compute the Hamiltonian for given displacement and volume fields.

        Args:
            displacements: (n_particles, dimension) array of displacements
            volume_fields: (n_particles,) array of volume fields

        Returns:
            Hamiltonian value
        """
        # Elastic term: (k/2) * sum_{n ∈ N(i)} |u_j - u_i|²
        elastic_energy = 0.0
        for i, neighbor_list in enumerate(self.neighbors):
            for j in neighbor_list:
                disp_diff = displacements[j] - displacements[i]
                elastic_energy += 0.5 * self.k * jnp.sum(disp_diff**2)

        # Volume term: (c/2) * sum_i v_i²
        volume_energy = 0.5 * self.c * jnp.sum(volume_fields**2)

        # Kappa term: (kappa/2) * sum_i |u_i|^2
        kappa_energy = 0.5 * self.kappa * jnp.sum(jnp.sum(displacements**2, axis=1))

        # Coupling term: (λ/2) * sum_i (v_i - b(∇·u)_i)²
        coupling_energy = 0.0
        for i, neighbor_list in enumerate(self.neighbors):
            # Compute divergence (∇·u)_i
            div_u = 0.0
            for j in neighbor_list:
                r_ij = self.lattice_positions[j] - self.lattice_positions[i]
                disp_diff = displacements[j] - displacements[i]
                div_u += jnp.dot(disp_diff, r_ij)

            coupling_term = volume_fields[i] - self.b * div_u
            coupling_energy += 0.5 * self.lambda_param * coupling_term**2

        return elastic_energy + volume_energy + kappa_energy + coupling_energy

    def _compute_driving_force(self, t: float) -> jnp.ndarray:
        """
        Compute the driving force for all sites at time t (volume driving).
        Returns:
            Array of driving forces for each site
        """
        driving_forces = jnp.zeros(self.n_particles)
        if self.driving_amplitude > 0 and self.driven_sites:
            for idx, site_idx in enumerate(self.driven_sites):
                phase = 0.0
                if self.driving_phases and idx < len(self.driving_phases):
                    phase = self.driving_phases[idx]
                if 0 <= site_idx < self.n_particles:
                    driving_forces = driving_forces.at[site_idx].set(
                        self.driving_amplitude
                        * jnp.sin(self.driving_frequency * t + phase)
                    )
        return driving_forces

    def _compute_displacement_driving_force(self, t: float) -> jnp.ndarray:
        """
        Compute the displacement driving force for all sites at time t.
        Returns:
            Array of shape (n_particles, dimension) with driving forces for each site
        """
        disp_driving_forces = jnp.zeros((self.n_particles, self.dimension))
        if self.displacement_driving_amplitude > 0 and self.displacement_driven_sites:
            for idx, site_idx in enumerate(self.displacement_driven_sites):
                phase = 0.0
                if self.displacement_driving_phases and idx < len(
                    self.displacement_driving_phases
                ):
                    phase = self.displacement_driving_phases[idx]
                direction = jnp.array(self.displacement_driving_directions[idx])
                if 0 <= site_idx < self.n_particles:
                    force = (
                        self.displacement_driving_amplitude
                        * jnp.sin(self.displacement_driving_frequency * t + phase)
                        * direction
                    )
                    disp_driving_forces = disp_driving_forces.at[site_idx].set(force)
        return disp_driving_forces

    def compute_derivatives(
        self, t: float, state: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """
        Compute derivatives using automatic differentiation from the Hamiltonian.
        """
        displacements, volume_fields = self._unwrap_state(state)

        # Create a function that takes displacements and volume_fields as separate arguments
        def hamiltonian_fn(disps, vols):
            return self.compute_hamiltonian(disps, vols)

        # Compute gradients using automatic differentiation
        grad_fn = jax.grad(hamiltonian_fn, argnums=(0, 1))
        grad_disps, grad_vols = grad_fn(displacements, volume_fields)

        # Apply overdamped dynamics: γ_u * u̇ = -∂H/∂u, γ_v * v̇ = -∂H/∂v
        disp_derivatives = -grad_disps / self.gamma_u
        vol_derivatives = -grad_vols / self.gamma_v

        # Add driving force to volume derivatives
        driving_forces = self._compute_driving_force(t)
        vol_derivatives = vol_derivatives + driving_forces

        # Add displacement driving force to displacement derivatives
        disp_driving_forces = self._compute_displacement_driving_force(t)
        disp_derivatives = disp_derivatives + disp_driving_forces

        return self.ravel_state(disp_derivatives, vol_derivatives)

    def return_state(self) -> jnp.ndarray:
        """Return the current state of the system."""
        return self.state

    def ravel_state(
        self, displacements: jnp.ndarray, volume_fields: jnp.ndarray
    ) -> jnp.ndarray:
        """Ravel displacements and volume fields into a single state vector."""
        return jnp.concatenate(
            [
                displacements.ravel(),  # Flatten displacements (shape: (N*d,))
                volume_fields.ravel(),  # Flatten volume fields (shape: (N,))
            ]
        )

    def _unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Unwraps the state into displacements and volume fields.
        Expects a flat state array of shape (N*d + N,)
        Returns displacements and volume fields as separate arrays.
        """
        disp_size = self.n_particles * self.dimension
        displacements = state[:disp_size].reshape(self.n_particles, self.dimension)
        volume_fields = state[disp_size:].reshape(
            self.n_particles,
        )

        return displacements, volume_fields

    def unwrap_state(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Unwraps the state into a dictionary of named components.
        Works across the entire trajectory.

        Args:
            state: State array of shape (state_dim,) for single state or (T, state_dim) for trajectory

        Returns:
            Dictionary with keys:
            - 'positions': (N, D) or (T, N, D) array for actual positions (lattice + displacement)
            - 'displacement_field': (N, D) or (T, N, D) array for displacement field
            - 'volumes': (N,) or (T, N) array for actual volumes (base + volume field)
            - 'volume_displacement_field': (N,) or (T, N) array for volume field
        """
        # Handle both single states and trajectories
        if len(state.shape) == 1:
            # Single state: (state_dim,)
            displacements, volume_fields = self._unwrap_state(state)
            positions = self.lattice_positions + displacements
            volumes = self.volumes + volume_fields
        else:
            # Trajectory: (T, state_dim)
            disp_size = self.n_particles * self.dimension
            displacements = state[:, :disp_size].reshape(
                -1, self.n_particles, self.dimension
            )
            volume_fields = state[:, disp_size:].reshape(-1, self.n_particles)
            positions = self.lattice_positions[None, :, :] + displacements  # (T, N, D)
            volumes = self.volumes[None, :] + volume_fields  # (T, N)

        return {
            "positions": positions,
            "displacement_field": displacements,
            "volumes": volumes,
            "volume_displacement_field": volume_fields,
        }

    def get_expected_state_shape(self) -> Tuple[int, ...]:
        """Get the expected shape of the state vector."""
        return (self.n_particles * self.dimension + self.n_particles,)

    def get_lattice_info(self) -> Dict[str, Any]:
        """Get information about the lattice structure."""
        return {
            "grid_shape": self.grid_shape,
            "lattice_spacing": self.lattice_spacing,
            "n_particles": self.n_particles,
            "dimension": self.dimension,
            "lattice_positions": self.lattice_positions,
            "neighbors": self.neighbors,
        }

    def to_networkx_graph(self):
        """
        Return the lattice as a NetworkX graph (nodes = sites, edges = neighbors).
        """
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self.n_particles))
        for i, nbrs in enumerate(self.neighbors):
            for j in nbrs:
                if i < j:  # Avoid duplicate edges
                    G.add_edge(i, j)
        return G


if __name__ == "__main__":
    # Test with the new configuration structure
    from ...config.schemas import DynamicsConfig

    config = DynamicsConfig(
        type="lattice_hamiltonian",
        n_particles=25,  # Will be overridden by grid_size
        dimension=2,
        parameters={
            "k": 1.0,
            "c": 1.0,
            "lambda": 0.5,
            "b": 1.0,
            "gamma_u": 1.0,
            "gamma_v": 1.0,
            "grid_type": "2d",
            "grid_size": 5,
            "lattice_spacing": 1.0,
            "seed": 42,
            "driving_amplitude": 0.1,
            "driving_frequency": 1.0,
            "driven_sites": [0, 1, 2],
            "driving_phases": [0.0, 0.5, 1.0],
            "displacement_driving_amplitude": 0.05,
            "displacement_driving_frequency": 2.0,
            "displacement_driven_sites": [3, 4],
            "displacement_driving_phases": [0.0, 0.5],
            "displacement_driving_directions": [[1.0, 0.0], [0.0, 1.0]],
            "kappa": 0.1,  # Added kappa parameter
        },
        initial_conditions={
            "displacement_range": [-0.1, 0.1],
            "volume_range": [-0.1, 0.1],
        },
    )

    system = LatticeHamiltonianSystem()
    system.initialize(config)
    initial_state = system.return_state()
    print("Initial State Shape:", initial_state.shape)
    print("Lattice Info:", system.get_lattice_info())

    # Compute derivative
    dstate = system.compute_derivatives(0.0, initial_state)
    print("Derivative Shape:", dstate.shape)

    # Test Hamiltonian computation
    displacements, volume_fields = system._unwrap_state(initial_state)
    hamiltonian = system.compute_hamiltonian(displacements, volume_fields)
    print("Initial Hamiltonian:", hamiltonian)
