"""Visualization module for dynamo-infer.

The matplotlib backend now uses the temporal graph visualizer by default.
If a graph is provided, edges will be shown; otherwise, only nodes are animated.
"""

from .matplotlib_visualization import MatplotlibVisualizer
from .blender_visualization import BlenderVisualizer


def create_visualizer(config):
    """Create a visualizer from configuration (expects config.backend or config.method).
    The matplotlib backend uses the temporal graph visualizer and will show edges if a graph is provided.
    """
    backend = getattr(config, "backend", None) or getattr(config, "method", None)
    if backend is None:
        raise ValueError("Visualization config must specify 'backend' or 'method'.")
    backend = backend.lower()
    if backend == "matplotlib":
        return MatplotlibVisualizer(config)
    elif backend == "blender":
        return BlenderVisualizer(config)
    else:
        raise ValueError(f"Unknown visualization backend: {backend}")


__all__ = [
    "MatplotlibVisualizer",
    "BlenderVisualizer",
    "create_visualizer",
]
