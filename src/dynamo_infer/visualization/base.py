"""Base visualization classes."""


class BaseVisualizer:
    """
    Abstract base class for all visualizers.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def visualize(self, pos, ori=None, save_path=None, **kwargs):
        """
        Visualize the trajectory (to be implemented by subclasses).
        """
        raise NotImplementedError("Subclasses must implement visualize()")
