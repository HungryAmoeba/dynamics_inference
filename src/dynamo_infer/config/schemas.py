"""Configuration schemas for dynamo-infer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class DynamicsConfig:
    """Configuration for dynamical systems."""
    
    type: str = "swarmalator"
    """Type of dynamical system (e.g., 'swarmalator', 'gravitation', 'ga_general')"""
    
    n_particles: int = 100
    """Number of particles in the system"""
    
    dimension: int = 3
    """Spatial dimension"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """System-specific parameters"""
    
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    """Initial condition specifications"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "n_particles": self.n_particles,
            "dimension": self.dimension,
            "parameters": self.parameters,
            "initial_conditions": self.initial_conditions,
        }


@dataclass
class TimeConfig:
    """Time configuration for simulation."""
    
    t0: float = 0.0
    """Start time"""
    
    t1: float = 10.0
    """End time"""
    
    dt: float = 0.01
    """Time step"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"t0": self.t0, "t1": self.t1, "dt": self.dt}


@dataclass
class SaveConfig:
    """Save configuration for simulation."""
    
    t0: Optional[float] = None
    """Save start time (defaults to simulation t0)"""
    
    t1: Optional[float] = None
    """Save end time (defaults to simulation t1)"""
    
    dt: Optional[float] = None
    """Save time step (defaults to simulation dt)"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"t0": self.t0, "t1": self.t1, "dt": self.dt}


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    
    solver: str = "Tsit5"
    """ODE solver type"""
    
    time: TimeConfig = field(default_factory=TimeConfig)
    """Time configuration"""
    
    saveat: SaveConfig = field(default_factory=SaveConfig)
    """Save configuration"""
    
    rtol: float = 1e-4
    """Relative tolerance"""
    
    atol: float = 1e-7
    """Absolute tolerance"""
    
    max_steps: Optional[int] = None
    """Maximum number of steps"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solver": self.solver,
            "time": self.time.to_dict(),
            "saveat": self.saveat.to_dict(),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    backend: str = "matplotlib"
    """Visualization backend ('matplotlib', 'plotly', 'blender', 'polyscope')"""
    
    animation: bool = True
    """Whether to create animations"""
    
    save_path: Optional[str] = None
    """Path to save visualizations"""
    
    fps: int = 30
    """Frames per second for animations"""
    
    quality: str = "high"
    """Rendering quality ('low', 'medium', 'high')"""
    
    show_trajectories: bool = False
    """Whether to show particle trajectories"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Backend-specific parameters"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "animation": self.animation,
            "save_path": self.save_path,
            "fps": self.fps,
            "quality": self.quality,
            "show_trajectories": self.show_trajectories,
            "parameters": self.parameters,
        }


@dataclass
class FeatureLibraryConfig:
    """Configuration for feature libraries."""
    
    type: str = "polynomial"
    """Feature library type ('polynomial', 'orthogonal_polynomial', 'custom')"""
    
    degree: int = 3
    """Polynomial degree"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Library-specific parameters"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "degree": self.degree,
            "parameters": self.parameters,
        }


@dataclass
class InferenceConfig:
    """Configuration for dynamics inference."""
    
    method: str = "GA_inference"
    """Inference method"""
    
    feature_library: FeatureLibraryConfig = field(default_factory=FeatureLibraryConfig)
    """Feature library configuration"""
    
    optimizer: str = "adam"
    """Optimizer type"""
    
    learning_rate: float = 1e-3
    """Learning rate"""
    
    epochs: int = 10000
    """Number of training epochs"""
    
    sparsity: float = 0.0
    """Sparsity regularization weight"""
    
    coupling_method: str = "dense"
    """Coupling method ('dense', 'gaussian', 'fixed', 'learn_fixed')"""
    
    differentiation_method: str = "finite_diff"
    """Method for computing derivatives ('finite_diff', 'savgol')"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Method-specific parameters"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "feature_library": self.feature_library.to_dict(),
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "sparsity": self.sparsity,
            "coupling_method": self.coupling_method,
            "differentiation_method": self.differentiation_method,
            "parameters": self.parameters,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    metrics: List[str] = field(default_factory=lambda: ["mse", "r2", "trajectory"])
    """Evaluation metrics to compute"""
    
    save_figures: bool = True
    """Whether to save evaluation figures"""
    
    save_model: bool = True
    """Whether to save the trained model"""
    
    save_results: bool = True
    """Whether to save evaluation results"""
    
    comparison_time_horizon: Optional[float] = None
    """Time horizon for trajectory comparison"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Evaluation-specific parameters"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "save_figures": self.save_figures,
            "save_model": self.save_model,
            "save_results": self.save_results,
            "comparison_time_horizon": self.comparison_time_horizon,
            "parameters": self.parameters,
        }