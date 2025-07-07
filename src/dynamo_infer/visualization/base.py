"""Base visualization classes."""

class DummyVisualizer:
    """Placeholder visualizer."""
    
    def __init__(self, config):
        self.config = config
        
    def visualize(self, trajectory, times, system):
        """Placeholder visualization."""
        print(f"Visualization with {self.config.backend} backend")
        return {"status": "completed", "backend": self.config.backend}