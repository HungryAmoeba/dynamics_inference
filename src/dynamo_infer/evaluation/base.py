"""Base evaluation classes."""

class DummyEvaluator:
    """Placeholder evaluator."""
    
    def __init__(self, config=None):
        self.config = config
        
    def evaluate(self, model, trajectory, times, system, output_dir):
        """Placeholder evaluation."""
        print("Running evaluation...")
        return {
            "metrics": {
                "mse": 0.001,
                "r2": 0.99,
            },
            "status": "completed"
        }