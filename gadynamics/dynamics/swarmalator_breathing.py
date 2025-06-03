import jax.numpy as jnp
import jax.random as jrandom
from gadynamics.dynamics.base import DynamicalSystem


class SwarmalatorBreathing(DynamicalSystem):
    def initialize(self, config, key=jrandom.PRNGKey(0)):

        self.dim = config.dimension
        self.N = config.num_agents
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.J = config.J
        self.R = config.R
        self.epsilon_a = config.epsilon_a
        self.epsilon_r = config.epsilon_r
        self.noise_strength = config.noise_strength

        # JAX random keys
        key, subkey1, subkey2, subkey3 = jrandom.split(key, 4)

        # Initialize positions in 3D space
        self.positions = jrandom.uniform(
            subkey1,
            (self.N, self.dim),
            minval=config.initial_conditions.position_range[0],
            maxval=config.initial_conditions.position_range[1],
        )

        # Initialize orientation (unit vector on a sphere)
        # theta = jrandom.uniform(
        #     subkey2, (self.N,),
        #     minval=config.initial_conditions.theta_range[0],
        #     maxval=config.initial_conditions.theta_range[1]
        # )
        # phi = jrandom.uniform(
        #     subkey3, (self.N,),
        #     minval=config.initial_conditions.phi_range[0],
        #     maxval=config.initial_conditions.phi_range[1]
        # )

        # self.orientations = jnp.stack([
        #     jnp.sin(theta) * jnp.cos(phi),
        #     jnp.sin(theta) * jnp.sin(phi),
        #     jnp.cos(theta)
        # ], axis=1)  # Shape: (N, 3)
        self.orientations = jrandom.normal(subkey2, (self.N, self.dim))
        self.orientations = self.orientations / jnp.linalg.norm(
            self.orientations, axis=1, keepdims=True
        )

        self.volumes = jrandom.uniform(
            subkey3, (self.N, 1), minval=0, maxval=jnp.pi * 2
        )

        self.velocities = jnp.zeros((self.N, self.dim))  # No self-propulsion initially

        print("Swarmalator model initialized.")

    # def return_state_unreshaped(self):
    #     """
    #     Return the current state (positions and orientations) as an array
    #     """
    #     return jnp.concat([self.positions, self.orientations])  # Shape: (N, D+3)

    def return_state(self):
        return jnp.concat(
            [self.positions, self.orientations, self.volumes], axis=1
        )  # Shape: (N, D+3+1)

        # return jnp.concat(
        #     [self.positions.ravel(), self.orientations.ravel()]
        # )  # Shape: (2ND,)

    # def compute_derivatives(self, t, state, args):
    #     """
    #     Compute the derivatives of the swarmalator system.
    #     Args:
    #         - t: current time (not used in this system)
    #         - state: current state (positions and orientations)
    #         - args: args (not used in this system)
    #     Returns:
    #         - dx_dt: Derivative of positions (N, D)
    #         - dsigma_dt: Derivative of orientations (N, D)
    #     Note: these are stacked on top of each other to form a single array of shape (2N, D)
    #     """
    #     # so we assume that state is given by
    #     # unpack state and then pass it off to compute_derivatives_unreshaped
    #     pos = state[: self.N * self.dim].reshape((self.N, self.dim))  # Shape: (N, D)
    #     ori = state[self.N * self.dim :].reshape((self.N, self.dim))
    #     state_reshape = jnp.concat([pos, ori], axis=0)  # Shape: (2N, D)
    #     deriv_unreshaped = self.compute_derivatives_unreshaped(state_reshape)
    #     # deriv_unreshaped_naive = self.compute_derivatives_unreshaped_naive(state_reshape)
    #     # deriv_unreshaped_naive = jnp.array(deriv_unreshaped)
    #     # import pdb; pdb.set_trace()
    #     # then flatten the derivatives to return them in the same shape as state
    #     dx_dt = deriv_unreshaped[: self.N]  # Shape: (N, D)
    #     dsigma_dt = deriv_unreshaped[self.N :]  # Shape: (N, D)

    #     return jnp.concat([dx_dt.ravel(), dsigma_dt.ravel()])  # Shape: (2ND,)s

    def compute_derivatives(self, t, state, args):
        N = self.N
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
        positions, orientations, vol = self.unwrap_state(state)  # Shape: (N, D), (N, D)
        positions = positions.reshape((N, self.dim))  # Ensure positions are (N, D)
        orientations = orientations.reshape(
            (N, self.dim)
        )  # Ensure orientations are (N, D)
        vol = vol.reshape((N, 1))  # Ensure volumes are (N, 1)

        # Pairwise differences and distances (CORRECTED HERE)
        pos_diffs = (
            positions[None, :, :] - positions[:, None, :]
        )  # Now (N, N, D) with x_j - x_i
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

        # # check if Ni and Nr are non-zero to avoid division by zero
        # if jnp.any(Ni == 0):
        #     raise ValueError("Some agents have no neighbors within the interaction radius R.")
        # if jnp.any(Nr == 0):
        #     raise ValueError("Some agents have no neighbors outside the interaction radius R.")

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

        # Volume dynamics -----------------------------------------------------------
        theta = vol.squeeze(-1)  # Shape: (N,)
        theta_diff = theta[None, :] - theta[:, None]  # Shape: (N, N)
        K = phase_coupling_strengths  # (N, N)
        dtheta_dt = jnp.sum(K * jnp.sin(theta_diff), axis=1)  # Shape: (N,)
        return jnp.concatenate([dx_dt, dsigma_dt, dtheta_dt[:, None]], axis=1)

        # return jnp.concatenate([dx_dt, dsigma_dt])

    def unwrap_state(self, state):
        """
        Unwrap the state into positions and orientations.
        Args:
            - state: The current state array (flattened)
        Returns:
            - positions: Array of shape (N, D) for positions
            - orientations: Array of shape (N, D) for orientations
        """
        if len(state.shape) == 2:
            state = state[None, :, :]
        pos = state[:, :, : self.dim]
        ori = state[:, :, self.dim : 2 * self.dim]
        vol = state[:, :, 2 * self.dim :]
        return pos, ori, vol
