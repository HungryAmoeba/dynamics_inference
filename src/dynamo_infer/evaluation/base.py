"""Base evaluation classes and metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import matplotlib.pyplot as plt

from ..config.schemas import EvaluationConfig


class BaseEvaluator(ABC):
    """Base class for evaluating inference models."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        
    @abstractmethod
    def evaluate(
        self, 
        model, 
        trajectory: jnp.ndarray,
        times: jnp.ndarray,
        system=None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Parameters
        ----------
        model : fitted inference model
            The trained model to evaluate
        trajectory : jnp.ndarray
            Ground truth trajectory data
        times : jnp.ndarray
            Time points
        system : optional
            The dynamical system (for additional metrics)
        output_dir : Path, optional
            Directory to save results
            
        Returns
        -------
        results : dict
            Evaluation results
        """
        pass


class DynamicsEvaluator(BaseEvaluator):
    """Evaluator for dynamics inference models."""
    
    def evaluate(
        self,
        model,
        trajectory: jnp.ndarray, 
        times: jnp.ndarray,
        system=None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of dynamics inference model.
        
        Parameters
        ----------
        model : DynamicsInferrer
            Trained inference model
        trajectory : jnp.ndarray
            Ground truth trajectory of shape (T, N, D)
        times : jnp.ndarray
            Time points of shape (T,)
        system : DynamicalSystem, optional
            Original dynamical system
        output_dir : Path, optional
            Directory to save evaluation results
            
        Returns
        -------
        results : dict
            Comprehensive evaluation results
        """
        results = {}
        
        # Predict derivatives
        pred_derivatives = model.predict(trajectory)
        
        # Compute ground truth derivatives if possible
        if system is not None:
            # Use system to compute true derivatives
            true_derivatives = []
            for i, state in enumerate(trajectory):
                deriv = system.compute_derivatives(times[i], state.flatten(), None)
                # Reshape to match trajectory format
                if hasattr(system, 'unwrap_state'):
                    pos, ori, *_ = system.unwrap_state(deriv.reshape(1, -1))
                    deriv_reshaped = jnp.concatenate([pos[0], ori[0]])
                else:
                    deriv_reshaped = deriv.reshape(state.shape)
                true_derivatives.append(deriv_reshaped)
            true_derivatives = jnp.stack(true_derivatives)
        else:
            # Use finite differences as approximation
            dt = times[1] - times[0] if len(times) > 1 else 1.0
            true_derivatives = jnp.gradient(trajectory, axis=0) / dt
        
        # Compute metrics
        metrics = self.compute_metrics(true_derivatives, pred_derivatives, model)
        results["metrics"] = metrics
        
        # Generate figures if requested
        if self.config.save_figures and output_dir is not None:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            figure_paths = self.generate_figures(
                true_derivatives, 
                pred_derivatives,
                trajectory,
                times,
                model,
                figures_dir
            )
            results["figures"] = figure_paths
        
        # Save model if requested
        if self.config.save_model and output_dir is not None:
            model_path = output_dir / "model.pkl"
            model.save_model(model_path)
            results["model_path"] = str(model_path)
            
        # Save equations if possible
        if hasattr(model, 'get_equations') and output_dir is not None:
            eq_path = output_dir / "equations.txt"
            try:
                equations = model.get_equations(symbolic=True)
                with open(eq_path, 'w') as f:
                    f.write("Learned Dynamics Equations\n")
                    f.write("=" * 40 + "\n")
                    for i, eq in enumerate(equations):
                        f.write(f"f_{i} = {eq}\n")
                results["equations_path"] = str(eq_path)
            except Exception as e:
                print(f"Could not save equations: {e}")
        
        # Save results summary if requested
        if self.config.save_results and output_dir is not None:
            self.save_results_summary(results, output_dir / "evaluation_summary.txt")
            
        return results
    
    def compute_metrics(
        self, 
        true_derivatives: jnp.ndarray,
        pred_derivatives: jnp.ndarray, 
        model
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        true_derivatives : jnp.ndarray
            Ground truth derivatives
        pred_derivatives : jnp.ndarray  
            Predicted derivatives
        model : inference model
            The trained model
            
        Returns
        -------
        metrics : dict
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Mean Squared Error
        mse = float(jnp.mean((true_derivatives - pred_derivatives) ** 2))
        metrics["mse"] = mse
        
        # Root Mean Squared Error
        metrics["rmse"] = float(jnp.sqrt(mse))
        
        # Mean Absolute Error
        mae = float(jnp.mean(jnp.abs(true_derivatives - pred_derivatives)))
        metrics["mae"] = mae
        
        # R² score
        ss_res = jnp.sum((true_derivatives - pred_derivatives) ** 2)
        ss_tot = jnp.sum((true_derivatives - jnp.mean(true_derivatives)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-10)))
        metrics["r2"] = r2
        
        # Component-wise metrics if model has grade information
        if hasattr(model, 'g_of_d'):
            g_of_d = jnp.array(model.g_of_d)
            for grade in jnp.unique(g_of_d):
                grade_idxs = jnp.where(g_of_d == grade)[0]
                if len(grade_idxs) > 0:
                    grade_mse = float(jnp.mean(
                        (true_derivatives[..., grade_idxs] - pred_derivatives[..., grade_idxs]) ** 2
                    ))
                    metrics[f"grade_{int(grade)}_mse"] = grade_mse
        
        return metrics
    
    def generate_figures(
        self,
        true_derivatives: jnp.ndarray,
        pred_derivatives: jnp.ndarray,
        trajectory: jnp.ndarray,
        times: jnp.ndarray,
        model,
        output_dir: Path
    ) -> Dict[str, str]:
        """
        Generate evaluation figures.
        
        Parameters
        ----------
        true_derivatives : jnp.ndarray
            Ground truth derivatives
        pred_derivatives : jnp.ndarray
            Predicted derivatives  
        trajectory : jnp.ndarray
            Full trajectory
        times : jnp.ndarray
            Time points
        model : inference model
            The trained model
        output_dir : Path
            Directory to save figures
            
        Returns
        -------
        figure_paths : dict
            Dictionary mapping figure names to file paths
        """
        figure_paths = {}
        
        # Trajectory comparison plot
        fig_path = self.plot_trajectory_comparison(
            true_derivatives, pred_derivatives, times, model, output_dir
        )
        figure_paths["trajectory_comparison"] = fig_path
        
        # Error over time plot
        fig_path = self.plot_error_over_time(
            true_derivatives, pred_derivatives, times, output_dir
        )
        figure_paths["error_over_time"] = fig_path
        
        # Parity plot (predicted vs true)
        fig_path = self.plot_parity(
            true_derivatives, pred_derivatives, output_dir
        )
        figure_paths["parity_plot"] = fig_path
        
        return figure_paths
    
    def plot_trajectory_comparison(
        self,
        true_derivatives: jnp.ndarray,
        pred_derivatives: jnp.ndarray,
        times: jnp.ndarray,
        model,
        output_dir: Path,
        num_components: int = 6,
        num_particles: int = 3
    ) -> str:
        """Plot comparison of true vs predicted derivatives."""
        T, N, D = true_derivatives.shape
        
        # Select subset for visualization
        particle_idxs = np.random.choice(N, min(num_particles, N), replace=False)
        component_idxs = np.random.choice(D, min(num_components, D), replace=False)
        
        fig, axes = plt.subplots(
            len(particle_idxs), len(component_idxs), 
            figsize=(4 * len(component_idxs), 3 * len(particle_idxs)),
            squeeze=False
        )
        
        for i, p in enumerate(particle_idxs):
            for j, c in enumerate(component_idxs):
                ax = axes[i, j]
                ax.plot(times, true_derivatives[:, p, c], 'b-', label='True', linewidth=2)
                ax.plot(times, pred_derivatives[:, p, c], 'r--', label='Predicted', linewidth=1)
                
                # Add grade information if available
                if hasattr(model, 'g_of_d'):
                    grade = model.g_of_d[c]
                    ax.set_title(f"Particle {p}, Component {c} (Grade {grade})")
                else:
                    ax.set_title(f"Particle {p}, Component {c}")
                    
                ax.set_xlabel("Time")
                ax.set_ylabel("Derivative")
                if i == 0 and j == 0:
                    ax.legend()
        
        plt.tight_layout()
        save_path = output_dir / "trajectory_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_error_over_time(
        self,
        true_derivatives: jnp.ndarray,
        pred_derivatives: jnp.ndarray,
        times: jnp.ndarray,
        output_dir: Path
    ) -> str:
        """Plot error over time."""
        errors = jnp.abs(true_derivatives - pred_derivatives)
        
        # Compute statistics over particles and components
        mean_error = jnp.mean(errors, axis=(1, 2))
        std_error = jnp.std(errors, axis=(1, 2))
        max_error = jnp.max(errors, axis=(1, 2))
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, mean_error, 'b-', label='Mean Error', linewidth=2)
        plt.fill_between(
            times, 
            mean_error - std_error, 
            mean_error + std_error, 
            alpha=0.3, color='blue', label='±1 Std'
        )
        plt.plot(times, max_error, 'r-', label='Max Error', linewidth=1)
        
        plt.xlabel("Time")
        plt.ylabel("Absolute Error")
        plt.title("Prediction Error Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = output_dir / "error_over_time.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_parity(
        self,
        true_derivatives: jnp.ndarray,
        pred_derivatives: jnp.ndarray,
        output_dir: Path
    ) -> str:
        """Plot predicted vs true values (parity plot)."""
        # Flatten for parity plot
        true_flat = true_derivatives.flatten()
        pred_flat = pred_derivatives.flatten()
        
        # Sample points for visualization (if too many)
        if len(true_flat) > 10000:
            idxs = np.random.choice(len(true_flat), 10000, replace=False)
            true_flat = true_flat[idxs]
            pred_flat = pred_flat[idxs]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(true_flat, pred_flat, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(jnp.min(true_flat), jnp.min(pred_flat))
        max_val = max(jnp.max(true_flat), jnp.max(pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel("True Derivatives")
        plt.ylabel("Predicted Derivatives")
        plt.title("Parity Plot: Predicted vs True")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make square
        plt.axis('equal')
        
        save_path = output_dir / "parity_plot.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def save_results_summary(self, results: Dict[str, Any], save_path: Path) -> None:
        """Save a summary of evaluation results."""
        with open(save_path, 'w') as f:
            f.write("Dynamics Inference Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Write metrics
            if "metrics" in results:
                f.write("Metrics:\n")
                f.write("-" * 20 + "\n")
                for metric, value in results["metrics"].items():
                    f.write(f"{metric}: {value:.6f}\n")
                f.write("\n")
            
            # Write file paths
            for key, value in results.items():
                if key.endswith("_path") or key == "figures":
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nEvaluation completed successfully.\n")