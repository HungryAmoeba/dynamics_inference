"""Swarmalator dynamics implementation."""

import jax.numpy as jnp
import jax.random as random
from typing import Tuple, Any, Dict
from ..base import InteractingParticleSystem
from ...config.schemas import DynamicsConfig


class Swarmalator(InteractingParticleSystem):
    """
    High-dimensional swarmalator system.

    Based on: Yadav, A., et al. (2024): Exotic swarming dynamics of high-dimensional
    swarmalators. Physical Review E, 109, 4, 044212.

    This implementation matches the carefully tested version in gadynamics/dynamics/swarmalator.py
    and uses only positions and orientations internally, like the gadynamics version.
    """

    def __init__(self):
        super().__init__()
        # Physics parameters
        self.alpha = 1.0  # Attraction exponent
        self.beta = 2.0  # Repulsion exponent
        self.gamma = 1.0  # Phase coupling exponent
        self.J = 1.0  # Orientation coupling strength
        self.R = 1.0  # Interaction radius
        self.epsilon_a = 1.0  # Attractive phase coupling strength
        self.epsilon_r = 1.0  # Repulsive phase coupling strength
        self.noise_strength = 0.0  # Noise strength

    def initialize(self, config: DynamicsConfig) -> None:
        """Initialize the swarmalator system."""
        super().initialize(config)

        # Get parameters from config
        params = config.parameters
        self.alpha = params.get("alpha", 1.0)
        self.beta = params.get("beta", 2.0)
        self.gamma = params.get("gamma", 1.0)
        self.J = params.get("J", 1.0)
        self.R = params.get("R", 1.0)
        self.epsilon_a = params.get("epsilon_a", 1.0)
        self.epsilon_r = params.get("epsilon_r", 1.0)
        self.noise_strength = params.get("noise_strength", 0.0)

        # Initialize random state
        seed = params.get("seed", 42)
        key = random.PRNGKey(seed)
        key1, key2 = random.split(key, 2)

        # Initialize positions randomly in a box
        pos_range = config.initial_conditions.get("position_range", [-2.5, 2.5])
        self.positions = random.uniform(
            key1,
            (self.n_particles, self.dimension),
            minval=pos_range[0],
            maxval=pos_range[1],
        )

        # Initialize orientations randomly on unit sphere
        self.orientations = random.normal(key2, (self.n_particles, self.dimension))
        norms = jnp.linalg.norm(self.orientations, axis=1, keepdims=True)
        self.orientations = self.orientations / norms

        # Create state in the format used by gadynamics: concatenate positions and orientations
        # This matches the gadynamics implementation exactly
        self.state = jnp.concatenate(
            [self.positions.flatten(), self.orientations.flatten()]  # (N*D,)  # (N*D,)
        )  # Shape: (2*N*D,)

        self.initialized = True

    def compute_derivatives(
        self, t: float, state: jnp.ndarray, args: Any = None
    ) -> jnp.ndarray:
        """Compute time derivatives for swarmalator dynamics."""
        # Reshape state to match gadynamics format: (2N, D)
        state_reshaped = state.reshape(2 * self.n_particles, self.dimension)

        # Compute derivatives using the unreshaped method (matches gadynamics)
        deriv_unreshaped = self._compute_derivatives_unreshaped(state_reshaped)

        # Removed JAX-incompatible asserts
        # assert not jnp.isnan(deriv_unreshaped).any(), "Derivatives contain NaNs"
        # assert not jnp.isinf(deriv_unreshaped).any(), "Derivatives contain Infs"

        # Flatten for return (matches gadynamics)
        return deriv_unreshaped.flatten()

    def _compute_derivatives_unreshaped(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute derivatives in the unreshaped format (2N, D).

        Args:
            state: State array of shape (2N, D) where first N rows are positions,
                   last N rows are orientations

        Returns:
            Derivatives array of shape (2N, D)
        """
        N = self.n_particles
        alpha, beta, gamma, J, R, epsilon_a, epsilon_r = (
            self.alpha,
            self.beta,
            self.gamma,
            self.J,
            self.R,
            self.epsilon_a,
            self.epsilon_r,
        )

        # Extract positions and orientations
        positions = state[:N, :]  # Shape: (N, D)
        orientations = state[N:, :]  # Shape: (N, D)

        # Pairwise differences and distances (CORRECTED - matches gadynamics)
        pos_diffs = (
            positions[None, :, :] - positions[:, None, :]
        )  # (N, N, D) with x_j - x_i
        min_distance = 1e-4
        distances = jnp.clip(jnp.linalg.norm(pos_diffs, axis=-1), min_distance, None)

        mask = 1 - jnp.eye(N)  # Exclude self-interaction (N, N)

        # Position dynamics --------------------------------------------------------
        orientation_dot = jnp.einsum("ij,kj->ik", orientations, orientations)  # (N, N)

        # Attraction/repulsion terms
        spatial_attraction = (1 + J * orientation_dot) / (distances**alpha)  # (N, N)
        spatial_repulsion = 1 / (distances**beta)  # (N, N)

        # Apply mask and compute forces (pos_diffs now correct, no sign flip needed)
        force_matrix = ((spatial_attraction - spatial_repulsion) * mask)[
            ..., None
        ] * pos_diffs
        dx_dt = jnp.sum(force_matrix, axis=1) / (N - 1)  # (N, D)

        # Orientation dynamics ------------------------------------------------------
        # Compute neighbor counts
        within_R = (distances < R).astype(jnp.float32)
        Ni = jnp.sum(within_R * mask, axis=1)  # j≠i within R
        Nr = (N - 1) - Ni  # j≠i outside R

        epsilon = 1e-9
        Ni_safe = Ni[:, None] + epsilon
        Nr_safe = Nr[:, None] + epsilon
        phase_coupling_strengths = jnp.where(
            within_R.astype(bool), epsilon_a / Ni_safe, -epsilon_r / Nr_safe
        ) / (distances**gamma)
        phase_coupling_strengths *= mask  # Exclude self-interaction

        # Orientation interaction term
        sigma_projection = orientations[None, :, :] - (
            orientation_dot[..., None] * orientations[:, None, :]
        )
        phase_interaction = phase_coupling_strengths[..., None] * sigma_projection
        dsigma_dt = jnp.sum(phase_interaction, axis=1)  # (N, D)

        return jnp.concatenate([dx_dt, dsigma_dt])

    def return_state(self) -> jnp.ndarray:
        """Return current state."""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        return self.state

    def unwrap_state(self, state: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Unwrap state into a dictionary of named components.
        This matches the gadynamics implementation.

        Args:
            state: Either a single state vector or a trajectory (T, state_dim)

        Returns:
            Dictionary with keys:
            - 'positions': (N, D) or (T, N, D) array - positions
            - 'orientations': (N, D) or (T, N, D) array - orientations
        """
        pos_size = self.n_particles * self.dimension

        # Handle both single states and trajectories
        if len(state.shape) == 1:
            # Single state vector
            positions = state[:pos_size].reshape(self.n_particles, self.dimension)
            orientations = state[pos_size:].reshape(self.n_particles, self.dimension)
        else:
            # Trajectory (T, state_dim)
            positions = state[:, :pos_size].reshape(
                -1, self.n_particles, self.dimension
            )
            orientations = state[:, pos_size:].reshape(
                -1, self.n_particles, self.dimension
            )

        return {"positions": positions, "orientations": orientations}

    def get_expected_state_shape(self) -> Tuple[int]:
        """Expected state shape: 2*N*D (positions + orientations)."""
        return (2 * self.n_particles * self.dimension,)
