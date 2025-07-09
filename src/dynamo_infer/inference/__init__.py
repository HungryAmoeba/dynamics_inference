"""Inference module for dynamo-infer."""

from .base import BaseInference, DynamicsInferrer
from .models import GA_DynamicsInference
from .feature_library import (
    BaseFeatureLibrary,
    MonomialPolynomialLibrary,
    OrthogonalPolynomialLibrary,
)
from ..config.schemas import InferenceConfig


def create_inferrer(config: InferenceConfig) -> DynamicsInferrer:
    """
    Create an inference model from configuration.
    
    Parameters
    ----------
    config : InferenceConfig
        Configuration for the inference method
        
    Returns
    -------
    inferrer : DynamicsInferrer
        Configured inference model
        
    Raises
    ------
    ValueError
        If method is not recognized
    """
    method = config.method.lower()
    
    if method == "ga_inference" or method == "geometric_algebra":
        return GA_DynamicsInference(config)
    else:
        raise ValueError(f"Unknown inference method: {config.method}")


__all__ = [
    "BaseInference",
    "DynamicsInferrer", 
    "GA_DynamicsInference",
    "BaseFeatureLibrary",
    "MonomialPolynomialLibrary",
    "OrthogonalPolynomialLibrary",
    "create_inferrer",
]