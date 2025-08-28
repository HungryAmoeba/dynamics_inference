# Dynamo-Infer: Modular Dynamics Simulation and Inference

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular framework for simulating, visualizing, and inferring dynamics of interacting systems. Dynamo-Infer provides a clean, scientific-computing-ready package that makes it easy to:

1. **Define** initial configurations and interaction rules
2. **Simulate** dynamics using robust ODE solvers
3. **Visualize** results with multiple backends
4. **Infer** dynamics from observational data
5. **Evaluate** and save results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamo-infer.git
cd dynamo-infer

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev,visualization,notebooks]"
```

### Basic Usage

```python
import dynamo_infer as di

# 1. Load configuration
config = di.load_config("configs/swarmalator.yaml")

# 2. Run the complete pipeline
results = di.run_full_pipeline(config)

# Or run individual steps:
system = di.create_system(config.dynamics)
simulator = di.create_simulator(system, config.simulation)
trajectory, times = simulator.run()
```

### Command Line Interface

```bash
# Run complete pipeline
dynamo-infer configs/swarmalator.yaml --output outputs/

# Run individual steps
dynamo-simulate configs/swarmalator.yaml
# (for inference, provide the trajectory file as a positional argument)
dynamo-infer-dynamics outputs/trajectory.npz
# (for evaluation, provide model and trajectory as positional arguments)
dynamo-evaluate outputs/model.pkl outputs/trajectory.npz
```

## 📁 Project Structure

```
dynamo-infer/
├── src/dynamo_infer/           # Main package
│   ├── config/                 # Configuration management
│   │   └── systems/           # Specific implementations
│   ├── simulation/             # Simulation engines
│   ├── visualization/          # Visualization backends
│   ├── inference/              # Dynamics inference methods
│   ├── evaluation/             # Performance evaluation
│   ├── utils/                  # Utility functions
│   └── workflow.py            # High-level orchestration
├── configs/                    # Configuration files
├── examples/                   # Example scripts and notebooks
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## 🔧 Configuration System

Dynamo-Infer uses a hierarchical YAML configuration system:

```yaml
# config.yaml
dynamics:
  type: "swarmalator"
  n_particles: 100
  dimension: 3
  parameters:
    J: 1.0  # Coupling strength
    K: 1.0  # Phase coupling
    seed: 42

simulation:
  solver: "Tsit5"
  time:
    t0: 0.0
    t1: 10.0
    dt: 0.01
  rtol: 1e-4
  atol: 1e-7

visualization:
  backend: "matplotlib"
  animation: true
  save_path: "outputs/animation.mp4"

inference:
  method: "GA_inference"
  feature_library:
    type: "orthogonal_polynomial"
    degree: 3
  epochs: 10000
  learning_rate: 1e-3

evaluation:
  metrics: ["mse", "r2", "trajectory"]
  save_figures: true
  save_model: true
```

## 🎯 Supported Systems

### Dynamical Systems
- **Swarmalators**: High-dimensional swarming with phase dynamics
- **Gravitational**: N-body gravitational interactions
- **Geometric Algebra**: General interacting systems using geometric algebra
- **Custom**: Easy extension for new systems

### Simulation Methods
- **Diffrax**: Modern JAX-based ODE solvers (Tsit5, Dopri5, etc.)
- **Adaptive stepping**: Automatic error control
- **GPU acceleration**: JAX-native computations

### Visualization Backends
- **Matplotlib**: Static plots and basic animations
- **Plotly**: Interactive 3D visualizations
- **Blender**: High-quality 3D renderings and animations
- **Polyscope**: Real-time 3D visualization

### Inference Methods
- **Geometric Algebra Inference**: Discover dynamics using GA representations
- **SINDy**: Sparse identification of nonlinear dynamics
- **Neural ODEs**: Deep learning approaches
- **Custom**: Extensible framework for new methods

## 📚 Examples

### Example 1: Swarmalator Simulation

```python
import dynamo_infer as di

# Create configuration
config = di.Config(
    dynamics=di.DynamicsConfig(
        type="swarmalator",
        n_particles=50,
        dimension=3,
        parameters={"J": 1.0, "K": 1.0}
    ),
    simulation=di.SimulationConfig(
        time=di.TimeConfig(t0=0, t1=10, dt=0.01)
    )
)

# Run simulation
system = di.create_system(config.dynamics)
simulator = di.create_simulator(system, config.simulation)
trajectory, times = simulator.run()

print(f"Simulated {trajectory.shape[0]} timesteps")
```

### Example 2: Complete Pipeline with Inference

```python
# Load configuration with all steps enabled
config = di.load_config("configs/complete_pipeline.yaml")

# Run everything
results = di.run_full_pipeline(
    config,
    output_dir="outputs/experiment_1",
    verbose=True
)

# Access results
trajectory = results["trajectory"]
model = results["model"]
evaluation = results["evaluation"]

print(f"MSE: {evaluation['metrics']['mse']:.6f}")
```

### Example 3: Custom System

```python
from dynamo_infer.dynamics.base import DynamicalSystem

class CustomOscillator(DynamicalSystem):
    def initialize(self, config):
        # Setup your system
        pass
        
    def compute_derivatives(self, t, state, args):
        # Define dynamics
        return derivatives

# Register your system
di.register_system("custom_oscillator", CustomOscillator)

# Use it in configs
config = di.DynamicsConfig(type="custom_oscillator", ...)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dynamo_infer

# Run specific test categories
pytest tests/test_dynamics.py
pytest tests/test_inference.py
```

## 📖 Documentation

- **[API Reference](docs/api/)**: Complete API documentation
- **[User Guide](docs/guide/)**: Step-by-step tutorials
- **[Examples](examples/)**: Jupyter notebooks and scripts
- **[Developer Guide](docs/dev/)**: Contributing and extending

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research from:
  - Yadav, A., et al. (2024): "Exotic swarming dynamics of high-dimensional swarmalators." Physical Review E
  - SINDy framework by Brunton et al.
  - JAX ecosystem for high-performance computing

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/dynamo-infer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/dynamo-infer/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for the scientific computing community**
