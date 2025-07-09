# Driving Force Functionality in LatticeHamiltonianSystem

This document explains how to use the periodic driving force functionality that has been added to the `LatticeHamiltonianSystem` class.

## Overview

The driving force functionality allows you to apply a periodic external force to specific lattice sites. The driving force is added to the volume field dynamics:

```
γ_v * v̇_i = -∂H/∂v_i + F_drive_i(t)
```

where `F_drive_i(t) = A * sin(ω*t)` for driven sites and `F_drive_i(t) = 0` for undriven sites.

## Configuration

To use the driving force functionality, add the following parameters to your `DynamicsConfig`:

```python
config = DynamicsConfig(
    type="lattice_hamiltonian",
    dimension=2,
    parameters={
        # ... other parameters ...
        
        # Driving force parameters
        "driving_amplitude": 0.5,  # Amplitude of driving force
        "driving_frequency": 2.0,  # Frequency in rad/s
        "driven_sites": [12],  # List of site indices to drive
    },
    # ... rest of config ...
)
```

### Parameters

- **`driving_amplitude`** (float): Amplitude of the sinusoidal driving force. Set to 0.0 to disable driving.
- **`driving_frequency`** (float): Angular frequency of the driving force in radians per second.
- **`driven_sites`** (list): List of site indices to apply the driving force to. Site indices are determined by the grid layout.

### Site Indexing

For a 2D grid, sites are indexed row by row:
```
For a 3x3 grid:
[0, 1, 2]
[3, 4, 5]  <- Site 4 is the center
[6, 7, 8]

For a 5x5 grid:
[ 0,  1,  2,  3,  4]
[ 5,  6,  7,  8,  9]
[10, 11, 12, 13, 14]  <- Site 12 is the center
[15, 16, 17, 18, 19]
[20, 21, 22, 23, 24]
```

## Examples

### Simple Example

```python
from src.dynamo_infer.config.schemas import DynamicsConfig, SimulationConfig, TimeConfig, SaveConfig
from src.dynamo_infer.dynamics.systems.lattice_hamiltonian import LatticeHamiltonianSystem
from src.dynamo_infer.simulation.factory import create_simulator

# Create configuration with driving force
config = DynamicsConfig(
    type="lattice_hamiltonian",
    dimension=2,
    parameters={
        "k": 1.0,
        "c": 1.0,
        "lambda": 0.5,
        "b": 1.0,
        "gamma_u": 1.0,
        "gamma_v": 1.0,
        "grid_type": "2d",
        "grid_size": 3,
        "lattice_spacing": 1.0,
        "seed": 42,
        "driving_amplitude": 1.0,
        "driving_frequency": 3.0,
        "driven_sites": [4],  # Drive center site
    },
    initial_conditions={
        "displacement_range": [-0.1, 0.1],
        "volume_range": [-0.1, 0.1],
    }
)

# Create simulation configuration
sim_config = SimulationConfig(
    solver="tsit5",
    time=TimeConfig(t0=0.0, t1=5.0, dt=0.01),
    saveat=SaveConfig(t0=0.0, t1=5.0, dt=0.01),
)

# Run simulation
system = LatticeHamiltonianSystem()
system.initialize(config)
simulator = create_simulator(system, sim_config)
trajectory, times = simulator.run()
```

### Multiple Driven Sites

You can drive multiple sites simultaneously:

```python
config = DynamicsConfig(
    # ... other parameters ...
    parameters={
        # ... other parameters ...
        "driving_amplitude": 0.5,
        "driving_frequency": 2.0,
        "driven_sites": [0, 4, 8],  # Drive corner sites and center
    },
)
```

### Different Driving Frequencies

To drive different sites with different frequencies, you can modify the `_compute_driving_force` method in the class or create a custom driving force function.

## Analysis

After running a simulation, you can analyze the results:

```python
# Unwrap the trajectory
states = system.unwrap_state(trajectory)
volumes = states["orientations"]  # Shape: (T, N)
displacements = states["positions"]  # Shape: (T, N, D)

# Plot volume field evolution for a specific site
import matplotlib.pyplot as plt
plt.plot(times, volumes[:, 4])  # Plot center site
plt.xlabel('Time')
plt.ylabel('Volume Field')
plt.title('Volume Field Evolution at Driven Site')
plt.show()

# Plot spatial distribution at a specific time
volume_grid = volumes[100].reshape(system.grid_shape)  # Reshape to grid
plt.imshow(volume_grid, cmap='RdBu_r', origin='lower')
plt.colorbar()
plt.title('Volume Field Distribution')
plt.show()
```

## Files

- **`src/dynamo_infer/dynamics/systems/lattice_hamiltonian.py`**: Modified to include driving force functionality
- **`test_driving_force.py`**: Comprehensive test script comparing driven vs undriven systems
- **`simple_driving_example.py`**: Simple example demonstrating the functionality

## Running the Examples

```bash
# Run the simple example
python simple_driving_example.py

# Run the comprehensive test
python test_driving_force.py
```

## Physics

The driving force adds energy to the system and can lead to interesting phenomena:

1. **Resonance**: When the driving frequency matches natural frequencies of the system
2. **Synchronization**: Neighboring sites may synchronize with the driven site
3. **Wave propagation**: The driving can create waves that propagate through the lattice
4. **Energy injection**: The system receives continuous energy input, leading to sustained oscillations

## Customization

To customize the driving force beyond simple sinusoidal forcing, you can modify the `_compute_driving_force` method in the `LatticeHamiltonianSystem` class. For example, to add different waveforms or time-dependent amplitudes. 