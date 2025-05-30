import jax.numpy as jnp
import jax.random as jrandom
import jax
from gadynamics.dynamics.base import DynamicalSystem


class GravitationalSystem(DynamicalSystem):
    def __init__(self):
        self.state = None
        self.dimension = None
        self.num_particles = None
        self.G = None
        self.masses = None
        self.rng = None

    def initialize(self, config):
        """
        Expects a config dictionary with keys:
          - 'dimension': 2 or 3 (default 3)
          - 'num_particles': number of particles (default 3)
          - 'G': gravitational constant (default 1.0)
          - 'initial_conditions': dict with:
              - 'position_range': [min, max] for initial positions (default [-1,1])
              - 'velocity_range': [min, max] for initial velocities (default [-0.1,0.1])
          - 'rng_key': (optional) a JAX PRNGKey; if not provided, a default key is used.
        """
        self.dimension = config.get("dimension", 3)
        self.num_particles = config.get("num_particles", 3)
        self.G = config.get("G", 1.0)
        pos_range = config.get("initial_conditions", {}).get(
            "position_range", [-1.0, 1.0]
        )
        vel_range = config.get("initial_conditions", {}).get(
            "velocity_range", [-0.1, 0.1]
        )
        # Use provided RNG key or a default one
        self.rng = (
            jrandom.PRNGKey(config.get("rng_key", 0))
            if isinstance(config.get("rng_key"), int)
            else config.get("rng_key", jrandom.PRNGKey(0))
        )

        # Create positions and velocities randomly
        positions = jrandom.uniform(
            self.rng,
            (self.num_particles, self.dimension),
            minval=pos_range[0],
            maxval=pos_range[1],
        )
        self.rng, rng_vel = jrandom.split(self.rng)
        velocities = jrandom.uniform(
            rng_vel,
            (self.num_particles, self.dimension),
            minval=vel_range[0],
            maxval=vel_range[1],
        )
        # For simplicity, we assign unit mass to every particle. This can be extended.
        self.masses = jnp.ones(self.num_particles)

        self.state = self.ravel_state(positions, velocities)
        # self.state = {'positions': positions, 'velocities': velocities}
        # self.state = jnp.concat

    def compute_derivatives(self, t, state, args):
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
        acceleration = jax.vmap(acceleration_on_particle)(
            jnp.arange(self.num_particles)
        )

        return self.ravel_state(velocities, acceleration)

    def return_state(self):
        return self.state

    def ravel_state(self, positions, velocities):
        return jnp.concat(
            [
                positions.ravel(),  # Flatten positions (shape: (N*d,))
                velocities.ravel(),  # Flatten velocities (shape: (N*d,))
            ]
        )

    def _unwrap_state(self, state):
        """
        Unwraps the state into positions and velocities.
        Expects a flat state array of shape (2*N*d,)
        Returns a dictionary with 'positions' and 'velocities'.
        """
        positions = state[: self.num_particles * self.dimension].reshape(
            self.num_particles, self.dimension
        )
        velocities = state[self.num_particles * self.dimension :].reshape(
            self.num_particles, self.dimension
        )

        return positions, velocities

    def unwrap_state(self, state):
        """
        Unwraps the state into positions and none since we don't have orientations or other state variables in this system.

        works across the entire trajectory
        """
        positions = state[:, : self.num_particles * self.dimension].reshape(
            -1, self.num_particles, self.dimension
        )

        return positions, None


if __name__ == "__main__":
    config = {
        "dimension": 3,  # working in 3D space
        "num_particles": 5,
        "G": 1.0,
        "initial_conditions": {
            "position_range": [-1.0, 1.0],
            "velocity_range": [-0.1, 0.1],
        },
        "rng_key": jrandom.PRNGKey(42),
    }

    system = GravitationalSystem()
    system.initialize(config)
    initial_state = system.return_state()
    print("Initial State:")
    print("Positions:\n", initial_state["positions"])
    print("Velocities:\n", initial_state["velocities"])

    # Compute derivative (state derivative, i.e., time derivatives of positions and velocities)
    dstate = system.derivative(initial_state)
    print("\nState Derivatives:")
    print("dPositions/dt:\n", dstate["positions"])
    print("dVelocities/dt:\n", dstate["velocities"])
