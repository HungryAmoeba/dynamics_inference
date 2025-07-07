"""Base inference classes."""

class DummyInferrer:
    """Placeholder inferrer."""
    
    def __init__(self, config):
        self.config = config
        
    def fit(self, trajectory, times):
        """Placeholder inference."""
        print(f"Running inference with {self.config.method}")
        return {"method": self.config.method, "status": "fitted"}
        
    def save_model(self, model, path):
        """Placeholder save."""
        print(f"Saving model to {path}")
        return True