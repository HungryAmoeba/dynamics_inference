import jax.numpy as jnp
import jax.random as jrandom
import jax
from typing import Tuple, Any, Dict
from ..base import DynamicalSystem
from ...config.schemas import DynamicsConfig


class GravitationalSystem(DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.G = None
        self.masses = None
        self.rng = None

    def initialize(self, config: DynamicsConfig) -> None:
        """
        Initialize the gravitational system with the provided configuration.

        Args:
            config: DynamicsConfig object specifying system parameters
        """
        super().initialize(config)

        # Get parameters from config
        params = config.parameters
        self.G = params.get("G", 1.0)

        # Get initial conditions
        initial_conditions = config.initial_conditions
        pos_range = initial_conditions.get("position_range", [-1.0, 1.0])
        vel_range = initial_conditions.get("velocity_range", [-0.1, 0.1])

        # Use provided RNG key or a default one
        seed = params.get("seed", 42)
        self.rng = jrandom.PRNGKey(seed)

        # Create positions and velocities randomly
        positions = jrandom.uniform(
            self.rng,
            (self.n_particles, self.dimension),
            minval=pos_range[0],
            maxval=pos_range[1],
        )
        self.rng, rng_vel = jrandom.split(self.rng)
        velocities = jrandom.uniform(
            rng_vel,
            (self.n_particles, self.dimension),
            minval=vel_range[0],
            maxval=vel_range[1],
        )

        # For simplicity, we assign unit mass to every particle. This can be extended.
        self.masses = jnp.ones(self.n_particles)

        self.state = self.ravel_state(positions, velocities)
        self.initialized = True

    def compute_derivatives(
        self, t: float, state: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """
        Computes the derivative d/dt state = { d(positions)/dt, d(velocities)/dt }.
        The derivative of the positions is the velocity.
        The derivative of the velocities is the gravitational acceleration computed from
        all pairwise interactions.
        """
        positions, velocities = self._unwrap_state(state)

        # Regularization to avoid singularities when particles are too close.
        eps = 1e-5

        def acceleration_on_particle(i):
            pos_i = positions[i]
            # Compute displacement vectors from particle i to all others
            disp = positions - pos_i  # shape (N, d)
            # Compute squared distances, add eps to avoid division by zero.
            dist_sq = jnp.sum(disp**2, axis=1) + eps
            # Newtonian acceleration: G * sum_{j != i} m_j * (r_j - r_i) / ||r_j - r_i||^3
            inv_dist3 = 1.0 / jnp.power(dist_sq, 1.5)
            # Exclude self-interaction by setting the factor for i == j to 0.
            inv_dist3 = inv_dist3.at[i].set(0.0)
            acc = self.G * jnp.sum(
                self.masses[:, None] * disp * inv_dist3[:, None], axis=0
            )
            return acc

        # Vectorize the acceleration computation for all particles.
        acceleration = jax.vmap(acceleration_on_particle)(jnp.arange(self.n_particles))

        return self.ravel_state(velocities, acceleration)

    def return_state(self) -> jnp.ndarray:
        """Return the current state of the system."""
        return self.state

    def ravel_state(
        self, positions: jnp.ndarray, velocities: jnp.ndarray
    ) -> jnp.ndarray:
        """Ravel positions and velocities into a single state vector."""
        return jnp.concatenate(
            [
                positions.ravel(),  # Flatten positions (shape: (N*d,))
                velocities.ravel(),  # Flatten velocities (shape: (N*d,))
            ]
        )

    def _unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Unwraps the state into positions and velocities.
        Expects a flat state array of shape (2*N*d,)
        Returns positions and velocities as separate arrays.
        """
        positions = state[: self.n_particles * self.dimension].reshape(
            self.n_particles, self.dimension
        )
        velocities = state[self.n_particles * self.dimension :].reshape(
            self.n_particles, self.dimension
        )

        return positions, velocities

    def unwrap_state(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Unwraps the state into a dictionary of named components.
        Works across the entire trajectory.

        Args:
            state: State array of shape (state_dim,) for single state or (T, state_dim) for trajectory

        Returns:
            Dictionary with keys:
            - 'positions': (N, D) or (T, N, D) array for positions
            - 'orientations': (N, D) or (T, N, D) array for velocities (renamed for GA compatibility)
        """
        # Handle both single states and trajectories
        if len(state.shape) == 1:
            # Single state: (state_dim,)
            positions = state[: self.n_particles * self.dimension].reshape(
                self.n_particles, self.dimension
            )
            velocities = state[self.n_particles * self.dimension :].reshape(
                self.n_particles, self.dimension
            )
        else:
            # Trajectory: (T, state_dim)
            positions = state[:, : self.n_particles * self.dimension].reshape(
                -1, self.n_particles, self.dimension
            )
            velocities = state[:, self.n_particles * self.dimension :].reshape(
                -1, self.n_particles, self.dimension
            )

        return {
            "positions": positions,
            "orientations": velocities,  # Use velocities as orientations for GA compatibility
        }

    def get_expected_state_shape(self) -> Tuple[int, ...]:
        """Get the expected shape of the state vector."""
        return (2 * self.n_particles * self.dimension,)


if __name__ == "__main__":
    # Test with the new configuration structure
    from ...config.schemas import DynamicsConfig

    config = DynamicsConfig(
        type="gravitation",
        n_particles=5,
        dimension=3,
        parameters={
            "G": 1.0,
            "seed": 42,
        },
        initial_conditions={
            "position_range": [-1.0, 1.0],
            "velocity_range": [-0.1, 0.1],
        },
    )

    system = GravitationalSystem()
    system.initialize(config)
    initial_state = system.return_state()
    print("Initial State Shape:", initial_state.shape)

    # Compute derivative
    dstate = system.compute_derivatives(0.0, initial_state)
    print("Derivative Shape:", dstate.shape)
