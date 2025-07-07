"""High-level workflow orchestration for dynamo-infer."""

from pathlib import Path
from typing import Optional, Dict, Any
import jax.numpy as jnp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from .config import Config, load_config
from .dynamics import create_system
from .simulation import create_simulator  
from .visualization import create_visualizer
from .inference import create_inferrer
from .evaluation import create_evaluator

console = Console()


def run_full_pipeline(
    config: Config,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    save_intermediate: bool = True
) -> Dict[str, Any]:
    """
    Run the complete 5-step dynamo-infer pipeline.
    
    Args:
        config: Complete configuration object
        output_dir: Directory to save outputs (defaults to config.output_dir)
        verbose: Whether to print progress information
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Dictionary containing all results from the pipeline
    """
    if output_dir is None:
        output_dir = config.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not verbose
    ) as progress:
        
        # Step 1: Create and initialize dynamical system
        task1 = progress.add_task("🔧 Creating dynamical system...", total=None)
        system = create_system(config.dynamics)
        progress.update(task1, description="✅ Dynamical system created")
        
        if verbose:
            console.print(f"[green]✓[/green] Created {config.dynamics.type} system with {config.dynamics.n_particles} particles")
        
        # Step 2: Run simulation
        task2 = progress.add_task("🚀 Running simulation...", total=None)
        simulator = create_simulator(system, config.simulation)
        trajectory, times = simulator.run()
        progress.update(task2, description="✅ Simulation completed")
        
        if verbose:
            console.print(f"[green]✓[/green] Simulation completed: {len(times)} timesteps over {times[-1]:.2f} time units")
        
        results["trajectory"] = trajectory
        results["times"] = times
        results["system"] = system
        
        if save_intermediate:
            traj_path = output_dir / "trajectory.npz"
            jnp.savez(traj_path, trajectory=trajectory, times=times)
            if verbose:
                console.print(f"[blue]💾[/blue] Trajectory saved to {traj_path}")
        
        # Step 3: Visualization (optional)
        if config.visualization is not None:
            task3 = progress.add_task("🎨 Creating visualizations...", total=None)
            visualizer = create_visualizer(config.visualization)
            vis_results = visualizer.visualize(trajectory, times, system)
            progress.update(task3, description="✅ Visualizations created")
            
            results["visualizations"] = vis_results
            
            if verbose:
                console.print(f"[green]✓[/green] Visualizations created using {config.visualization.backend}")
        
        # Step 4: Dynamics inference (optional)
        if config.inference is not None:
            task4 = progress.add_task("🧠 Running dynamics inference...", total=None)
            inferrer = create_inferrer(config.inference)
            model = inferrer.fit(trajectory, times)
            progress.update(task4, description="✅ Inference completed")
            
            results["model"] = model
            results["inferrer"] = inferrer
            
            if verbose:
                console.print(f"[green]✓[/green] Dynamics inference completed using {config.inference.method}")
            
            if save_intermediate:
                model_path = output_dir / "model.pkl"
                inferrer.save_model(model, model_path)
                if verbose:
                    console.print(f"[blue]💾[/blue] Model saved to {model_path}")
        
        # Step 5: Evaluation (optional)
        if config.evaluation is not None and "model" in results:
            task5 = progress.add_task("📊 Evaluating results...", total=None)
            evaluator = create_evaluator(config.evaluation)
            eval_results = evaluator.evaluate(
                model=results["model"],
                trajectory=trajectory,
                times=times,
                system=system,
                output_dir=output_dir
            )
            progress.update(task5, description="✅ Evaluation completed")
            
            results["evaluation"] = eval_results
            
            if verbose:
                console.print(f"[green]✓[/green] Evaluation completed")
                for metric, value in eval_results.get("metrics", {}).items():
                    console.print(f"  {metric}: {value:.6f}")
    
    # Save complete results
    if save_intermediate:
        summary_path = output_dir / "pipeline_summary.yaml"
        save_pipeline_summary(results, summary_path)
        if verbose:
            console.print(f"[blue]💾[/blue] Pipeline summary saved to {summary_path}")
    
    if verbose:
        console.print(f"\n[bold green]🎉 Pipeline completed successfully![/bold green]")
        console.print(f"[blue]📁 All outputs saved to: {output_dir}[/blue]")
    
    return results


def run_pipeline_from_config_file(
    config_path: str,
    output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run pipeline from a configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Output directory (optional)
        **kwargs: Additional arguments passed to run_full_pipeline
        
    Returns:
        Pipeline results
    """
    config = load_config(config_path)
    
    if output_dir is not None:
        config.output_dir = Path(output_dir)
    
    return run_full_pipeline(config, **kwargs)


def save_pipeline_summary(results: Dict[str, Any], save_path: Path) -> None:
    """Save a summary of pipeline results."""
    import yaml
    
    summary = {
        "completed_steps": list(results.keys()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Add trajectory info
    if "trajectory" in results:
        trajectory = results["trajectory"]
        summary["trajectory"] = {
            "shape": list(trajectory.shape),
            "duration": float(results["times"][-1]),
            "n_timesteps": len(results["times"]),
        }
    
    # Add model info
    if "model" in results:
        summary["inference"] = {
            "method": type(results["model"]).__name__,
            "completed": True,
        }
    
    # Add evaluation metrics
    if "evaluation" in results:
        summary["evaluation"] = results["evaluation"].get("metrics", {})
    
    with open(save_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)