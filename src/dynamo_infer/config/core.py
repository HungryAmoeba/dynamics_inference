"""Core configuration management."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import DictConfig, OmegaConf

from .schemas import (
    DynamicsConfig,
    SimulationConfig,
    VisualizationConfig,
    InferenceConfig,
    EvaluationConfig,
    TimeConfig,
    SaveConfig,
    FeatureLibraryConfig,
)


class Config:
    """Main configuration container for dynamo-infer workflows."""

    def __init__(
        self,
        dynamics: DynamicsConfig,
        simulation: SimulationConfig,
        visualization: Optional[VisualizationConfig] = None,
        inference: Optional[InferenceConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        output_dir: str = "./outputs",
        **kwargs,
    ):
        self.dynamics = dynamics
        self.simulation = simulation
        self.visualization = visualization
        self.inference = inference
        self.evaluation = evaluation
        self.output_dir = Path(output_dir)

        # Store any additional configuration
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Extract main sections
        dynamics = DynamicsConfig(**config_dict.get("dynamics", {}))

        # --- FIX: ensure nested config objects are constructed ---
        sim_dict = config_dict.get("simulation", {})
        if "time" in sim_dict and isinstance(sim_dict["time"], dict):
            sim_dict["time"] = TimeConfig(**sim_dict["time"])
        if "saveat" in sim_dict and isinstance(sim_dict["saveat"], dict):
            sim_dict["saveat"] = SaveConfig(**sim_dict["saveat"])
        simulation = SimulationConfig(**sim_dict)

        # Optional sections
        visualization = None
        if "visualization" in config_dict:
            visualization = VisualizationConfig(**config_dict["visualization"])

        inference = None
        if "inference" in config_dict:
            # --- FIX: ensure nested config objects are constructed ---
            inf_dict = config_dict["inference"]
            if "feature_library" in inf_dict and isinstance(
                inf_dict["feature_library"], dict
            ):
                inf_dict["feature_library"] = FeatureLibraryConfig(
                    **inf_dict["feature_library"]
                )
            inference = InferenceConfig(**inf_dict)

        evaluation = None
        if "evaluation" in config_dict:
            evaluation = EvaluationConfig(**config_dict["evaluation"])

        # Other keys
        other_keys = {
            k: v
            for k, v in config_dict.items()
            if k
            not in {
                "dynamics",
                "simulation",
                "visualization",
                "inference",
                "evaluation",
            }
        }

        return cls(
            dynamics=dynamics,
            simulation=simulation,
            visualization=visualization,
            inference=inference,
            evaluation=evaluation,
            **other_keys,
        )

    @classmethod
    def from_hydra_config(cls, hydra_config: DictConfig) -> "Config":
        """Create Config from Hydra DictConfig."""
        return cls.from_dict(OmegaConf.to_container(hydra_config, resolve=True))

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {}

        if self.dynamics:
            result["dynamics"] = self.dynamics.to_dict()
        if self.simulation:
            result["simulation"] = self.simulation.to_dict()
        if self.visualization:
            result["visualization"] = self.visualization.to_dict()
        if self.inference:
            result["inference"] = self.inference.to_dict()
        if self.evaluation:
            result["evaluation"] = self.evaluation.to_dict()

        # Add other attributes
        for key, value in self.__dict__.items():
            if key not in {
                "dynamics",
                "simulation",
                "visualization",
                "inference",
                "evaluation",
            }:
                if isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value

        return result


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
