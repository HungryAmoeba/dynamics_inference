"""Evaluation module for dynamo-infer."""

from .base import BaseEvaluator, DynamicsEvaluator
from ..config.schemas import EvaluationConfig
from typing import Optional


def create_evaluator(config: Optional[EvaluationConfig] = None) -> BaseEvaluator:
    """
    Create an evaluator from configuration.
    
    Parameters
    ----------
    config : EvaluationConfig, optional
        Configuration for evaluation. If None, uses defaults.
        
    Returns
    -------
    evaluator : BaseEvaluator
        Configured evaluator
    """
    return DynamicsEvaluator(config)


class Evaluator(BaseEvaluator):
    """Alias for backward compatibility."""
    pass


__all__ = [
    "BaseEvaluator",
    "DynamicsEvaluator",
    "Evaluator",
    "create_evaluator",
]