"""Swarmalator dynamics implementation."""

import jax.numpy as jnp
import jax.random as random
from typing import Tuple, Any
from ..base import InteractingParticleSystem
from ...config.schemas import DynamicsConfig


class Swarmalator(InteractingParticleSystem):
    """
    High-dimensional swarmalator system.
    
    Based on: Yadav, A., et al. (2024): Exotic swarming dynamics of high-dimensional 
    swarmalators. Physical Review E, 109, 4, 044212.
    """
    
    def __init__(self):
        super().__init__()
        self.J = 1.0  # Coupling strength
        self.K = 1.0  # Phase coupling
        self.w = None  # Natural frequencies
        self.phi = None  # Phases
        
    def initialize(self, config: DynamicsConfig) -> None:
        """Initialize the swarmalator system."""
        super().initialize(config)
        
        # Get parameters from config
        params = config.parameters
        self.J = params.get("J", 1.0)
        self.K = params.get("K", 1.0)
        
        # Initialize random state
        seed = params.get("seed", 42)
        key = random.PRNGKey(seed)
        key1, key2, key3, key4 = random.split(key, 4)
        
        # Initialize positions randomly in a box
        box_size = params.get("box_size", 5.0)
        self.positions = box_size * (random.uniform(key1, (self.n_particles, self.dimension)) - 0.5)
        
        # Initialize orientations randomly on unit sphere
        self.orientations = random.normal(key2, (self.n_particles, self.dimension))
        norms = jnp.linalg.norm(self.orientations, axis=1, keepdims=True)
        self.orientations = self.orientations / norms
        
        # Initialize natural frequencies
        freq_std = params.get("frequency_std", 0.1)
        self.w = random.normal(key3, (self.n_particles,)) * freq_std
        
        # Initialize phases
        self.phi = random.uniform(key4, (self.n_particles,)) * 2 * jnp.pi
        
        # Combine state
        self.state = jnp.concatenate([
            self.positions.flatten(),
            self.orientations.flatten(),
            self.phi
        ])
        
        self.initialized = True
    
    def compute_derivatives(self, t: float, state: jnp.ndarray, args: Any = None) -> jnp.ndarray:
        """Compute time derivatives for swarmalator dynamics."""
        # Extract components from state
        pos_size = self.n_particles * self.dimension
        ori_size = self.n_particles * self.dimension
        
        positions = state[:pos_size].reshape(self.n_particles, self.dimension)
        orientations = state[pos_size:pos_size + ori_size].reshape(self.n_particles, self.dimension)
        phases = state[pos_size + ori_size:]
        
        # Compute pairwise distances and phase differences
        pos_diff = positions[:, None, :] - positions[None, :, :]  # (N, N, D)
        dist = jnp.linalg.norm(pos_diff, axis=2)  # (N, N)
        
        phase_diff = phases[:, None] - phases[None, :]  # (N, N)
        
        # Compute position derivatives (swarming dynamics)
        # F_ij = (1 + J * cos(phi_i - phi_j)) * r_ij / |r_ij|
        force_factor = 1.0 + self.J * jnp.cos(phase_diff)
        
        # Avoid division by zero
        safe_dist = jnp.where(dist > 1e-10, dist, 1e-10)
        force_direction = pos_diff / safe_dist[:, :, None]
        
        # Sum forces from all other particles
        forces = jnp.sum(force_factor[:, :, None] * force_direction, axis=1)
        
        # Position derivatives
        dpos_dt = forces
        
        # Orientation derivatives (alignment with position velocity)
        dori_dt = dpos_dt
        # Normalize to keep on unit sphere
        norms = jnp.linalg.norm(dori_dt, axis=1, keepdims=True)
        dori_dt = jnp.where(norms > 1e-10, dori_dt / norms, dori_dt)
        
        # Phase derivatives
        # dphi_dt = w_i + K * sum_j sin(phi_j - phi_i) / |r_ij|
        phase_coupling = jnp.sum(jnp.sin(phase_diff) / safe_dist, axis=1)
        dphi_dt = self.w + self.K * phase_coupling
        
        # Combine derivatives
        return jnp.concatenate([
            dpos_dt.flatten(),
            dori_dt.flatten(), 
            dphi_dt
        ])
    
    def return_state(self) -> jnp.ndarray:
        """Return current state."""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        return self.state
    
    def unwrap_state(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Unwrap state into positions, orientations, and phases.
        
        Returns:
            (positions, orientations, phases) where:
            - positions: (N, D) array
            - orientations: (N, D) array  
            - phases: (N,) array
        """
        pos_size = self.n_particles * self.dimension
        ori_size = self.n_particles * self.dimension
        
        positions = state[:pos_size].reshape(self.n_particles, self.dimension)
        orientations = state[pos_size:pos_size + ori_size].reshape(self.n_particles, self.dimension)
        phases = state[pos_size + ori_size:]
        
        return positions, orientations, phases
    
    def get_expected_state_shape(self) -> Tuple[int]:
        """Expected state shape: positions + orientations + phases."""
        return (2 * self.n_particles * self.dimension + self.n_particles,)