#!/usr/bin/env python3
"""Simulate the Lattice Hamiltonian System dynamics."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.dynamo_infer.config.schemas import DynamicsConfig, SimulationConfig
from src.dynamo_infer.dynamics.factory import create_system
from src.dynamo_infer.simulation.factory import create_simulator


def simulate_lattice_dynamics():
    """Simulate lattice Hamiltonian dynamics."""
    print("Simulating Lattice Hamiltonian Dynamics")
    print("=" * 50)

    # Create dynamics configuration
    dynamics_config = DynamicsConfig(
        type="lattice_hamiltonian",
        n_particles=25,
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
            "displacement_range": [-0.2, 0.2],  # Larger initial displacements
            "volume_range": [-0.2, 0.2],  # Larger initial volume fields
        },
    )

    # Create simulation configuration
    from src.dynamo_infer.config.schemas import TimeConfig, SaveConfig

    simulation_config = SimulationConfig(
        solver="Tsit5",
        time=TimeConfig(t0=0.0, t1=5.0, dt=0.01),
        saveat=SaveConfig(t0=0.0, t1=5.0, dt=0.1),
        rtol=1e-4,
        atol=1e-7,
        max_steps=10000,
    )

    # Create system and simulator
    system = create_system(dynamics_config)
    simulator = create_simulator(system, simulation_config)

    # Run simulation
    print("Running simulation...")
    trajectory, times = simulator.run()
    print(f"Simulation completed!")
    print(f"Time points: {len(times)}")
    print(f"Solution shape: {trajectory.shape}")

    # Return a solution-like object for downstream code
    class Solution:
        pass

    solution = Solution()
    solution.y = trajectory.T  # (state_dim, T)
    solution.t = times
    return system, solution


def analyze_trajectory(system, solution):
    """Analyze the simulation trajectory."""
    print("\nAnalyzing Trajectory")
    print("=" * 30)

    # Extract trajectory data
    trajectory = solution.y.T  # Shape: (T, state_dim)
    times = solution.t

    # Unwrap trajectory
    unwrapped = system.unwrap_state(trajectory)
    displacements = unwrapped["positions"]  # Shape: (T, N, D)
    volume_fields = unwrapped["orientations"]  # Shape: (T, N)

    # Compute energy over time
    energies = []
    for t in range(len(times)):
        energy = system.compute_hamiltonian(displacements[t], volume_fields[t])
        energies.append(energy)

    energies = jnp.array(energies)

    # Compute statistics
    print(f"Initial energy: {energies[0]:.6f}")
    print(f"Final energy: {energies[-1]:.6f}")
    print(f"Energy reduction: {100 * (energies[0] - energies[-1]) / energies[0]:.2f}%")

    # Compute displacement statistics
    disp_norms = jnp.linalg.norm(displacements, axis=2)  # Shape: (T, N)
    vol_rms = jnp.sqrt(jnp.mean(volume_fields**2, axis=1))  # Shape: (T,)

    print(f"Initial displacement norm: {jnp.mean(disp_norms[0]):.6f}")
    print(f"Final displacement norm: {jnp.mean(disp_norms[-1]):.6f}")
    print(f"Initial volume RMS: {vol_rms[0]:.6f}")
    print(f"Final volume RMS: {vol_rms[-1]:.6f}")

    return times, energies, displacements, volume_fields, disp_norms, vol_rms


def plot_results(
    system, times, energies, displacements, volume_fields, disp_norms, vol_rms
):
    """Plot simulation results."""
    print("\nCreating plots...")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Energy over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(times, energies, "b-", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Hamiltonian Energy")
    ax1.set_title("Energy Evolution")
    ax1.grid(True, alpha=0.3)

    # 2. Displacement norm over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, jnp.mean(disp_norms, axis=1), "r-", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mean Displacement Norm")
    ax2.set_title("Displacement Evolution")
    ax2.grid(True, alpha=0.3)

    # 3. Volume RMS over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, vol_rms, "g-", linewidth=2)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Volume Field RMS")
    ax3.set_title("Volume Field Evolution")
    ax3.grid(True, alpha=0.3)

    # 4. Initial state visualization
    ax4 = plt.subplot(2, 3, 4)
    lattice_positions = system.lattice_positions
    initial_displacements = displacements[0]
    initial_volumes = volume_fields[0]

    # Plot original and displaced positions
    ax4.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        c="blue",
        s=50,
        alpha=0.6,
        label="Original",
    )
    displaced_positions = lattice_positions + initial_displacements
    ax4.scatter(
        displaced_positions[:, 0],
        displaced_positions[:, 1],
        c="red",
        s=50,
        alpha=0.6,
        label="Displaced",
    )

    # Draw arrows
    for i in range(len(lattice_positions)):
        ax4.arrow(
            lattice_positions[i, 0],
            lattice_positions[i, 1],
            initial_displacements[i, 0],
            initial_displacements[i, 1],
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
            alpha=0.5,
        )

    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title("Initial State")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal")

    # 5. Final state visualization
    ax5 = plt.subplot(2, 3, 5)
    final_displacements = displacements[-1]
    final_volumes = volume_fields[-1]

    # Plot original and displaced positions
    ax5.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        c="blue",
        s=50,
        alpha=0.6,
        label="Original",
    )
    displaced_positions = lattice_positions + final_displacements
    ax5.scatter(
        displaced_positions[:, 0],
        displaced_positions[:, 1],
        c="red",
        s=50,
        alpha=0.6,
        label="Displaced",
    )

    # Draw arrows
    for i in range(len(lattice_positions)):
        ax5.arrow(
            lattice_positions[i, 0],
            lattice_positions[i, 1],
            final_displacements[i, 0],
            final_displacements[i, 1],
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
            alpha=0.5,
        )

    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_title("Final State")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect("equal")

    # 6. Volume field evolution
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(
        lattice_positions[:, 0],
        lattice_positions[:, 1],
        c=final_volumes,
        cmap="viridis",
        s=100,
    )
    ax6.set_xlabel("X")
    ax6.set_ylabel("Y")
    ax6.set_title("Final Volume Fields")
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect("equal")

    # Add colorbar
    plt.colorbar(scatter, ax=ax6, label="Volume Field")

    plt.tight_layout()
    plt.savefig("lattice_hamiltonian_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Create animation frames
    create_animation_frames(system, times, displacements, volume_fields)


def create_animation_frames(system, times, displacements, volume_fields):
    """Create animation frames for the simulation."""
    print("Creating animation frames...")

    # Select time points for animation (every 10th frame)
    frame_indices = np.arange(0, len(times), 10)
    if frame_indices[-1] != len(times) - 1:
        frame_indices = np.append(frame_indices, len(times) - 1)

    lattice_positions = system.lattice_positions

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, frame_idx in enumerate(frame_indices):
        ax1.clear()
        ax2.clear()

        t = times[frame_idx]
        disp = displacements[frame_idx]
        vol = volume_fields[frame_idx]

        # Plot displacements
        ax1.scatter(
            lattice_positions[:, 0],
            lattice_positions[:, 1],
            c="blue",
            s=50,
            alpha=0.6,
            label="Original",
        )
        displaced_positions = lattice_positions + disp
        ax1.scatter(
            displaced_positions[:, 0],
            displaced_positions[:, 1],
            c="red",
            s=50,
            alpha=0.6,
            label="Displaced",
        )

        # Draw arrows
        for j in range(len(lattice_positions)):
            ax1.arrow(
                lattice_positions[j, 0],
                lattice_positions[j, 1],
                disp[j, 0],
                disp[j, 1],
                head_width=0.05,
                head_length=0.05,
                fc="black",
                ec="black",
                alpha=0.5,
            )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(f"Displacements at t={t:.2f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Plot volume fields
        scatter = ax2.scatter(
            lattice_positions[:, 0],
            lattice_positions[:, 1],
            c=vol,
            cmap="viridis",
            s=100,
        )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"Volume Fields at t={t:.2f}")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        plt.colorbar(scatter, ax=ax2, label="Volume Field")
        plt.tight_layout()

        # Save frame
        plt.savefig(f"animation_frames/frame_{i:03d}.png", dpi=150, bbox_inches="tight")

    plt.close()
    print(f"Created {len(frame_indices)} animation frames")


if __name__ == "__main__":
    # Create animation frames directory
    import os

    os.makedirs("animation_frames", exist_ok=True)

    # Run simulation
    system, solution = simulate_lattice_dynamics()

    # Analyze results
    times, energies, displacements, volume_fields, disp_norms, vol_rms = (
        analyze_trajectory(system, solution)
    )

    # Plot results
    plot_results(
        system, times, energies, displacements, volume_fields, disp_norms, vol_rms
    )

    print("\nSimulation and analysis completed!")
