#!/usr/bin/env python3
"""Test script for the Lattice Hamiltonian System."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.dynamo_infer.config.schemas import DynamicsConfig
from src.dynamo_infer.dynamics.factory import create_system


def test_2d_lattice():
    """Test 2D lattice Hamiltonian system."""
    print("Testing 2D Lattice Hamiltonian System")
    print("=" * 50)

    # Create configuration for 2D system
    config = DynamicsConfig(
        type="lattice_hamiltonian",
        n_particles=25,  # Will be overridden by grid_size
        dimension=2,
        parameters={
            "k": 1.0,  # Elastic constant
            "c": 1.0,  # Volume stiffness
            "lambda": 0.5,  # Coupling parameter
            "b": 1.0,  # Coupling coefficient
            "gamma_u": 1.0,  # Displacement damping
            "gamma_v": 1.0,  # Volume damping
            "grid_type": "2d",
            "grid_size": 5,  # 5x5 grid
            "lattice_spacing": 1.0,
            "seed": 42,
        },
        initial_conditions={
            "displacement_range": [-0.1, 0.1],
            "volume_range": [-0.1, 0.1],
        },
    )

    # Create and initialize system
    system = create_system(config)

    # Get initial state
    initial_state = system.return_state()
    displacements, volume_fields = system._unwrap_state(initial_state)

    print(f"Grid shape: {system.grid_shape}")
    print(f"Number of particles: {system.n_particles}")
    print(f"State shape: {initial_state.shape}")
    print(f"Expected shape: {system.get_expected_state_shape()}")

    # Compute initial Hamiltonian
    hamiltonian = system.compute_hamiltonian(displacements, volume_fields)
    print(f"Initial Hamiltonian: {hamiltonian:.6f}")

    # Compute derivatives
    derivatives = system.compute_derivatives(0.0, initial_state)
    disp_derivs, vol_derivs = system._unwrap_state(derivatives)

    print(f"Displacement derivatives norm: {jnp.linalg.norm(disp_derivs):.6f}")
    print(f"Volume derivatives norm: {jnp.linalg.norm(vol_derivs):.6f}")

    # Test neighbor structure
    lattice_info = system.get_lattice_info()
    print(f"Number of neighbors for site 0: {len(lattice_info['neighbors'][0])}")
    print(f"Neighbors of site 0: {lattice_info['neighbors'][0]}")

    return system, initial_state


def test_3d_lattice():
    """Test 3D lattice Hamiltonian system."""
    print("\nTesting 3D Lattice Hamiltonian System")
    print("=" * 50)

    # Create configuration for 3D system
    config = DynamicsConfig(
        type="lattice_hamiltonian",
        n_particles=125,  # Will be overridden by grid_size
        dimension=3,
        parameters={
            "k": 1.0,
            "c": 1.0,
            "lambda": 0.5,
            "b": 1.0,
            "gamma_u": 1.0,
            "gamma_v": 1.0,
            "grid_type": "3d",
            "grid_size": 5,  # 5x5x5 grid
            "lattice_spacing": 1.0,
            "seed": 42,
        },
        initial_conditions={
            "displacement_range": [-0.1, 0.1],
            "volume_range": [-0.1, 0.1],
        },
    )

    # Create and initialize system
    system = create_system(config)

    # Get initial state
    initial_state = system.return_state()
    displacements, volume_fields = system._unwrap_state(initial_state)

    print(f"Grid shape: {system.grid_shape}")
    print(f"Number of particles: {system.n_particles}")
    print(f"State shape: {initial_state.shape}")
    print(f"Expected shape: {system.get_expected_state_shape()}")

    # Compute initial Hamiltonian
    hamiltonian = system.compute_hamiltonian(displacements, volume_fields)
    print(f"Initial Hamiltonian: {hamiltonian:.6f}")

    # Test neighbor structure
    lattice_info = system.get_lattice_info()
    print(f"Number of neighbors for site 0: {len(lattice_info['neighbors'][0])}")
    print(f"Neighbors of site 0: {lattice_info['neighbors'][0]}")

    return system, initial_state


def visualize_2d_lattice(system, state):
    """Visualize the 2D lattice state."""
    displacements, volume_fields = system._unwrap_state(state)
    lattice_positions = system.lattice_positions

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot lattice positions with displacements
    ax1.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        c="blue",
        s=50,
        alpha=0.6,
        label="Original positions",
    )

    # Plot displaced positions
    displaced_positions = lattice_positions + displacements
    ax1.scatter(
        displaced_positions[:, 0],
        displaced_positions[:, 1],
        c="red",
        s=50,
        alpha=0.6,
        label="Displaced positions",
    )

    # Draw arrows from original to displaced positions
    for i in range(len(lattice_positions)):
        ax1.arrow(
            lattice_positions[i, 0],
            lattice_positions[i, 1],
            displacements[i, 0],
            displacements[i, 1],
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
            alpha=0.5,
        )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Lattice Displacements")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot volume fields
    scatter = ax2.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        c=volume_fields,
        cmap="viridis",
        s=100,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Volume Fields")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label="Volume Field")

    plt.tight_layout()
    plt.savefig("lattice_hamiltonian_2d.png", dpi=150, bbox_inches="tight")
    plt.show()


def test_energy_conservation():
    """Test that the Hamiltonian decreases over time (overdamped dynamics)."""
    print("\nTesting Energy Conservation (Overdamped Dynamics)")
    print("=" * 50)

    # Create system
    config = DynamicsConfig(
        type="lattice_hamiltonian",
        n_particles=25,
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
        },
        initial_conditions={
            "displacement_range": [-0.1, 0.1],
            "volume_range": [-0.1, 0.1],
        },
    )

    system = create_system(config)
    state = system.return_state()
    displacements, volume_fields = system._unwrap_state(state)

    # Compute initial energy
    initial_energy = system.compute_hamiltonian(displacements, volume_fields)
    print(f"Initial energy: {initial_energy:.6f}")

    # Compute derivatives
    derivatives = system.compute_derivatives(0.0, state)
    disp_derivs, vol_derivs = system._unwrap_state(derivatives)

    # Compute energy change rate
    energy_rate = (
        jnp.sum(disp_derivs * disp_derivs) * system.gamma_u
        + jnp.sum(vol_derivs * vol_derivs) * system.gamma_v
    )

    print(f"Energy change rate: {energy_rate:.6f}")
    print(f"Energy should decrease: {energy_rate > 0}")

    # Test small time step
    dt = 0.01
    new_displacements = displacements + disp_derivs * dt
    new_volume_fields = volume_fields + vol_derivs * dt

    new_energy = system.compute_hamiltonian(new_displacements, new_volume_fields)
    print(f"Energy after small step: {new_energy:.6f}")
    print(f"Energy decreased: {new_energy < initial_energy}")


if __name__ == "__main__":
    # Test 2D system
    system_2d, state_2d = test_2d_lattice()

    # Test 3D system
    system_3d, state_3d = test_3d_lattice()

    # Test energy conservation
    test_energy_conservation()

    # Visualize 2D system
    try:
        visualize_2d_lattice(system_2d, state_2d)
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\nAll tests completed successfully!")
