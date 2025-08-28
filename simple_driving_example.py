#!/usr/bin/env python3
"""
Simple example demonstrating the driving force functionality in LatticeHamiltonianSystem.

This script shows how to:
1. Create a lattice system with a periodic driving force
2. Configure which site to drive
3. Run a simulation and observe the response
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.dynamo_infer.config.schemas import (
    DynamicsConfig,
    SimulationConfig,
    TimeConfig,
    SaveConfig,
)
from src.dynamo_infer.dynamics.systems.lattice_hamiltonian import (
    LatticeHamiltonianSystem,
)
from src.dynamo_infer.simulation.factory import create_simulator


def main():
    """Simple example of driving force functionality."""

    # Create a 3x3 lattice with driving force on the center site
    config = DynamicsConfig(
        type="lattice_hamiltonian",
        dimension=2,
        parameters={
            # System parameters
            "k": 1.0,  # Elastic constant
            "c": 1.0,  # Volume stiffness
            "lambda": 0.5,  # Coupling parameter
            "b": 1.0,  # Coupling coefficient
            "gamma_u": 1.0,  # Displacement damping
            "gamma_v": 1.0,  # Volume damping
            # Grid parameters
            "grid_type": "2d",
            "grid_size": 3,  # 3x3 grid (9 sites total)
            "lattice_spacing": 1.0,
            "seed": 42,
            # Driving force parameters
            "driving_amplitude": 1.0,  # Strong driving force
            "driving_frequency": 3.0,  # Fast oscillation
            "driven_sites": [4],  # Drive the center site (index 4 in 3x3 grid)
        },
        initial_conditions={
            "displacement_range": [-0.1, 0.1],
            "volume_range": [-0.1, 0.1],
        },
    )

    # Create simulation configuration
    t_span = (0.0, 5.0)  # 5 seconds
    n_steps = 500  # 500 time steps
    dt = (t_span[1] - t_span[0]) / n_steps

    sim_config = SimulationConfig(
        solver="tsit5",
        time=TimeConfig(t0=t_span[0], t1=t_span[1], dt=dt),
        saveat=SaveConfig(t0=t_span[0], t1=t_span[1], dt=dt),
        rtol=1e-6,
        atol=1e-8,
    )

    print("Creating and initializing lattice system...")
    system = LatticeHamiltonianSystem()
    system.initialize(config)

    print(f"System has {system.n_particles} particles in a {system.grid_shape} grid")
    print(
        f"Driving site {config.parameters['driven_sites']} with amplitude {config.parameters['driving_amplitude']}"
    )
    print(f"and frequency {config.parameters['driving_frequency']} rad/s")

    # Create simulator and run
    print("Running simulation...")
    simulator = create_simulator(system, sim_config)
    trajectory, times = simulator.run()

    # Analyze results
    print("Analyzing results...")

    # Unwrap the trajectory
    states = system.unwrap_state(trajectory)
    volumes = states["orientations"]  # Shape: (T, N)

    # Plot the volume field evolution for all sites
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for site in range(9):
        axes[site].plot(times, volumes[:, site], "b-", linewidth=2)
        axes[site].set_title(f"Site {site}")
        axes[site].set_xlabel("Time")
        axes[site].set_ylabel("Volume Field")
        axes[site].grid(True)

        # Highlight the driven site
        if site == 4:  # Center site
            axes[site].set_title(
                f"Site {site} (DRIVEN)", color="red", fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig("simple_driving_example.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot spatial distribution at different times
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    time_indices = [
        0,
        len(times) // 4,
        len(times) // 2,
        3 * len(times) // 4,
        len(times) - 1,
    ]

    for i, t_idx in enumerate(time_indices):
        volume_grid = volumes[t_idx].reshape(3, 3)
        im = axes[i].imshow(volume_grid, cmap="RdBu_r", origin="lower")
        axes[i].set_title(f"t = {times[t_idx]:.2f}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")

        # Mark the driven site
        axes[i].plot(
            1, 1, "ko", markersize=15, markeredgecolor="white", markeredgewidth=3
        )

        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("simple_spatial_response.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Simulation complete!")
    print("Generated plots:")
    print("- simple_driving_example.png: Volume field evolution for all sites")
    print("- simple_spatial_response.png: Spatial distribution at different times")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"Final volume field at driven site: {volumes[-1, 4]:.4f}")
    print(f"Final volume field at corner site: {volumes[-1, 0]:.4f}")
    print(f"Volume field range: [{volumes.min():.4f}, {volumes.max():.4f}]")


if __name__ == "__main__":
    main()
