# Chiral Metamaterial Integration with dynamo_infer

This document describes how to use the chiral metamaterial system that has been integrated into the dynamo_infer pipeline.

## Overview

The `ChiralMetamaterial` class simulates a 3D lattice of nodes connected by springs, where each node has:
- **Position** (x, y, z coordinates)
- **Rotation angle** θ (scalar rotation)
- **Position velocities** (dx/dt, dy/dt, dz/dt)
- **Angular velocity** (dθ/dt)

The system includes:
- Elastic forces from deformed edges
- Rotational spring forces
- Damping
- Optional external forcing and velocity constraints

## Configuration

### Basic Configuration

```yaml
dynamics:
  type: "chiral_metamaterial"
  n_particles: 0  # Will be set automatically by grid size
  dimension: 3
  
  parameters:
    # Physical parameters
    L: 1.0          # nominal bar length
    k_e: 1.0e4      # bar-spring stiffness
    r: 0.2          # winding radius (affects chirality)
    k_r: 1.0e6      # rotational spring stiffness
    gamma: 1.0      # translational damping
    gamma_t: 0.1    # rotational damping
    dt: 1.0e-3      # internal timestep
    
    # Grid parameters
    grid_type: "3d"
    grid_size: [8, 8, 8]  # Creates 8×8×8 lattice
    
    # Random seed
    seed: 42
    
  initial_conditions:
    position_noise: 0.01    # noise level for initial positions
    angle_noise: 0.01       # noise level for initial angles
    velocity_noise: 0.01    # noise level for initial velocities
```

### Optional Forcing/Constraints

The system supports external forcing and velocity constraints:

#### Velocity Constraints
```yaml
parameters:
  forcing:
    type: "velocity_constraint"
    v_top_z: -0.25      # z-velocity for top layer nodes
    t_release: 20.0     # time when constraints are released
```

#### Constant Rotation
```yaml
parameters:
  forcing:
    type: "constant_rotation"
    rot_rate: 0.15      # rotation rate for all nodes
    t_release: 20.0     # time when rotation stops
```

## Usage Examples

### 1. Running from Configuration File

```bash
# Create a configuration file (see configs/chiral_metamaterial.yaml)
python -m src.dynamo_infer.workflow configs/chiral_metamaterial.yaml
```

### 2. Programmatic Usage

```python
from src.dynamo_infer.config.schemas import DynamicsConfig
from src.dynamo_infer.dynamics import create_system
from src.dynamo_infer.workflow import run_full_pipeline

# Create configuration
config = DynamicsConfig(
    type="chiral_metamaterial",
    dimension=3,
    parameters={
        "L": 1.0, "k_e": 1e4, "r": 0.2, "k_r": 1e6,
        "gamma": 1.0, "gamma_t": 0.1, "dt": 1e-3,
        "grid_type": "3d", "grid_size": [6, 6, 6],
        "seed": 42
    },
    initial_conditions={
        "position_noise": 0.01,
        "angle_noise": 0.01,
        "velocity_noise": 0.01
    }
)

# Create and use system
system = create_system(config)
state = system.return_state()
derivatives = system.compute_derivatives(0.0, state)

# Or run full pipeline
results = run_full_pipeline(config)
```

### 3. State Structure

The system state includes all degrees of freedom:

```python
# Unwrap state to get components
unwrapped = system.unwrap_state(trajectory)

# Available components:
positions = unwrapped["positions"]          # (T, N, 3) - node positions
angles = unwrapped["angles"]                # (T, N) - rotation angles
position_velocities = unwrapped["position_velocities"]  # (T, N, 3)
angular_velocities = unwrapped["angular_velocities"]    # (T, N)
orientations = unwrapped["orientations"]    # (T, N, 3) - computed from angles
```

## Physics

The system energy consists of:

1. **Elastic Energy**: `(k_e/2) * Σ(|r_ij| - L_ij)²`
   - Where `L_ij = L - r*(θ_i + θ_j)` (chirality coupling)

2. **Rotational Energy**: `(k_r/2) * Σ(θ_i)²`

The dynamics follow:
- `m * d²r_i/dt² = -∂H/∂r_i - γ * dr_i/dt + F_ext_i`
- `I * d²θ_i/dt² = -∂H/∂θ_i - γ_t * dθ_i/dt + T_ext_i`

With optional external forces `F_ext` and torques `T_ext`.

## Visualization

The system integrates with dynamo_infer's visualization:

```yaml
visualization:
  backend: "matplotlib"
  animation: true
  save_path: "chiral_metamaterial.mp4"
  parameters:
    title: "3D Chiral Metamaterial"
    cmap: "plasma"        # Colors nodes by rotation angle
    show_colorbar: true
```

## Inference

The system works with dynamo_infer's inference methods:

```yaml
inference:
  method: "GA_inference"  # Geometric Algebra inference
  Gn: 3                   # 3D geometric algebra
  feature_library:
    type: "polynomial"
    degree: 2
```

## Testing

Run the integration test:

```bash
python test_chiral_metamaterial_integration.py
```

This will verify:
- System creation and initialization
- Configuration file loading
- Pipeline execution
- State unwrapping and derivatives computation

## Files Modified/Added

1. **`src/dynamo_infer/dynamics/systems/chiral_metamaterial.py`** - Main system implementation
2. **`src/dynamo_infer/dynamics/factory.py`** - Added system registration
3. **`configs/chiral_metamaterial.yaml`** - Example configuration
4. **`test_chiral_metamaterial_integration.py`** - Integration tests

## Key Features

- **Full dynamo_infer compatibility**: Inherits from `DynamicalSystem` base class
- **Graph-based topology**: Uses NetworkX for flexible lattice structures
- **JAX-based computation**: Fast, differentiable physics simulation
- **Configurable forcing**: Support for various boundary conditions
- **State unwrapping**: Clean interface for accessing physical quantities
- **Visualization ready**: Integrates with matplotlib-based animation
- **Inference compatible**: Works with geometric algebra and other inference methods

The chiral metamaterial system is now fully integrated and can be used like any other dynamical system in the dynamo_infer pipeline!