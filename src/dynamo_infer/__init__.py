"""
Dynamo-Infer: A modular framework for simulating, visualizing, and inferring dynamics of interacting systems.

This package provides a clean, modular interface for:
1. Defining initial configurations and interaction rules
2. Simulating dynamics  
3. Visualizing results
4. Running dynamics inference
5. Evaluating and saving results

Example:
    >>> import dynamo_infer as di
    >>> 
    >>> # 1. Define system configuration
    >>> config = di.load_config("swarmalator.yaml")
    >>> system = di.create_system(config)
    >>> 
    >>> # 2. Simulate dynamics
    >>> simulator = di.create_simulator(system, config.simulation)
    >>> trajectory = simulator.run()
    >>> 
    >>> # 3. Visualize
    >>> visualizer = di.create_visualizer(config.visualization)
    >>> visualizer.animate(trajectory)
    >>> 
    >>> # 4. Run inference
    >>> inferrer = di.create_inferrer(config.inference)
    >>> model = inferrer.fit(trajectory)
    >>> 
    >>> # 5. Evaluate and save
    >>> evaluator = di.create_evaluator()
    >>> results = evaluator.evaluate(model, trajectory)
    >>> evaluator.save_results(results, "output/")
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core API imports - these provide the main 5-step workflow
from .config import load_config, Config
from .dynamics import create_system, DynamicalSystem
from .simulation import create_simulator, Simulator  
from .visualization import create_visualizer, Visualizer
from .inference import create_inferrer, DynamicsInferrer
from .evaluation import create_evaluator, Evaluator

# Convenience imports for advanced users
from .dynamics.systems import *
from .inference.models import *
from .utils import *

# High-level workflow function
from .workflow import run_full_pipeline

__all__ = [
    # Core workflow functions
    "load_config",
    "create_system", 
    "create_simulator",
    "create_visualizer", 
    "create_inferrer",
    "create_evaluator",
    "run_full_pipeline",
    
    # Core classes
    "Config",
    "DynamicalSystem",
    "Simulator", 
    "Visualizer",
    "DynamicsInferrer", 
    "Evaluator",
]