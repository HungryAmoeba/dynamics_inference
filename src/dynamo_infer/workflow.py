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
    save_intermediate: bool = True,
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
        disable=not verbose,
    ) as progress:

        # Step 1: Create and initialize dynamical system
        task1 = progress.add_task("🔧 Creating dynamical system...", total=None)
        system = create_system(config.dynamics)
        progress.update(task1, description="✅ Dynamical system created")

        if verbose:
            console.print(
                f"[green]✓[/green] Created {config.dynamics.type} system with {config.dynamics.n_particles} particles"
            )

        # Step 2: Run simulation
        task2 = progress.add_task("🚀 Running simulation...", total=None)
        simulator = create_simulator(system, config.simulation)
        trajectory, times = simulator.run()
        progress.update(task2, description="✅ Simulation completed")

        if verbose:
            console.print(
                f"[green]✓[/green] Simulation completed: {len(times)} timesteps over {times[-1]:.2f} time units"
            )

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

            # Debug: print trajectory shape
            if verbose:
                console.print(
                    f"[blue]DEBUG:[/blue] Trajectory shape: {trajectory.shape}"
                )

            # Handle visualization save path - save to output directory if specified
            save_path = config.visualization.save_path
            if save_path is not None:
                # If save_path is relative, make it relative to output_dir
                if not Path(save_path).is_absolute():
                    save_path = output_dir / Path(save_path).name
                else:
                    # If absolute, still save to output_dir with same filename
                    save_path = output_dir / Path(save_path).name

                # Fallback: if no extension, add .mp4
                import os

                if not os.path.splitext(str(save_path))[1]:
                    save_path = str(save_path) + ".mp4"

                if verbose:
                    console.print(
                        f"[blue]DEBUG:[/blue] Visualization will be saved to: {save_path}"
                    )

            # Extract positions and orientations from trajectory using system's unwrap_state method
            # Handle different system types that may have different state structures
            try:
                # Pass the full trajectory to unwrap_state to get trajectory-shaped outputs
                unwrapped = system.unwrap_state(trajectory)

                if verbose:
                    console.print(
                        f"[blue]DEBUG:[/blue] Unwrapped state has {len(unwrapped)} components: {list(unwrapped.keys())}"
                    )

                # Extract positions and volumes from the dictionary
                positions = unwrapped.get("positions")
                volumes = unwrapped.get("volumes")

                # Convert to numpy arrays if needed
                def to_numpy(arr):
                    import numpy as np

                    if arr is not None and hasattr(arr, "device_buffer"):
                        return np.array(arr)
                    return arr

                positions = to_numpy(positions)
                volumes = to_numpy(volumes)

                # If system is LatticeHamiltonianSystem, get the lattice graph
                graph = None
                if system.__class__.__name__ == "LatticeHamiltonianSystem":
                    graph = system.to_networkx_graph()

                # Pass visualization parameters as kwargs
                vis_params = dict(getattr(config.visualization, "parameters", {}))

                if positions is not None:
                    # For lattice hamiltonian, use volumes as node_sizes
                    if volumes is not None:
                        if verbose:
                            console.print(
                                f"[blue]DEBUG:[/blue] Visualizing positions with node sizes from volumes, shapes: {positions.shape}, {volumes.shape}"
                            )
                        vis_results = visualizer.visualize(
                            positions,
                            save_path=save_path,
                            graph=graph,
                            node_sizes=volumes,
                            **vis_params,
                        )
                    else:
                        if verbose:
                            console.print(
                                f"[blue]DEBUG:[/blue] Visualizing positions only, shape: {positions.shape}"
                            )
                        vis_results = visualizer.visualize(
                            positions, save_path=save_path, graph=graph, **vis_params
                        )
                else:
                    # Fallback: pass trajectory directly
                    if verbose:
                        console.print(
                            f"[blue]DEBUG:[/blue] Fallback: passing trajectory directly, shape: {trajectory.shape}"
                        )

                    # Reshape trajectory to (T, N, D) format for visualization
                    if len(trajectory.shape) == 2:
                        # trajectory is (T, state_dim), need to reshape to (T, N, D)
                        T = trajectory.shape[0]
                        state_dim = trajectory.shape[1]
                        N = system.n_particles
                        D = state_dim // N  # Assuming state_dim = N * D

                        if state_dim == N * D:
                            trajectory_reshaped = trajectory.reshape(T, N, D)
                            if verbose:
                                console.print(
                                    f"[blue]DEBUG:[/blue] Reshaped trajectory to (T, N, D): {trajectory_reshaped.shape}"
                                )
                            vis_results = visualizer.visualize(
                                trajectory_reshaped, save_path=save_path
                            )
                        else:
                            if verbose:
                                console.print(
                                    f"[yellow]⚠️[/yellow] Cannot reshape trajectory {trajectory.shape} to (T, N, D) format"
                                )
                            vis_results = None
                    else:
                        vis_results = visualizer.visualize(
                            trajectory, save_path=save_path
                        )

            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠️[/yellow] Visualization failed: {e}")
                    console.print(
                        f"[yellow]⚠️[/yellow] Falling back to trajectory-only visualization"
                    )
                # Fallback: pass trajectory directly
                vis_results = visualizer.visualize(trajectory, save_path=save_path)

            progress.update(task3, description="✅ Visualizations created")

            results["visualizations"] = vis_results

            if verbose:
                console.print(
                    f"[green]✓[/green] Visualizations created using {config.visualization.backend}"
                )

        # Step 4: Dynamics inference (optional)
        if config.inference is not None:
            task4 = progress.add_task("🧠 Running dynamics inference...", total=None)
            try:
                if verbose:
                    console.print(
                        f"[blue]DEBUG:[/blue] Starting inference with method: {config.inference.method}"
                    )
                    console.print(
                        f"[blue]DEBUG:[/blue] Trajectory shape for inference: {trajectory.shape}"
                    )

                # Get unwrapped state data
                unwrapped = system.unwrap_state(trajectory)

                if verbose:
                    console.print(
                        f"[blue]DEBUG:[/blue] Unwrapped state has {len(unwrapped)} components: {list(unwrapped.keys())}"
                    )

                # For GA inference, use the promote_to_GA utility
                if config.inference.method.lower() == "ga_inference":
                    from dynamo_infer.utils import promote_to_GA

                    # Get Gn from config or use default
                    Gn = getattr(config.inference, "Gn", 3)

                    if verbose:
                        console.print(
                            f"[blue]DEBUG:[/blue] Using promote_to_GA with Gn={Gn}"
                        )

                    # Promote unwrapped data to GA format
                    trajectory_for_inference = promote_to_GA(Gn, unwrapped, times)

                    if verbose:
                        console.print(
                            f"[blue]DEBUG:[/blue] GA inference: promoted to shape {trajectory_for_inference.shape}"
                        )
                else:
                    # For other inference methods, use positions only
                    positions = unwrapped.get("positions")

                    if positions is not None:
                        # Use positions for inference (most common case)
                        trajectory_for_inference = positions
                        if verbose:
                            console.print(
                                f"[blue]DEBUG:[/blue] Using positions for inference, shape: {trajectory_for_inference.shape}"
                            )
                    else:
                        # Fallback: use full trajectory
                        trajectory_for_inference = trajectory
                        if verbose:
                            console.print(
                                f"[blue]DEBUG:[/blue] Using full trajectory for inference, shape: {trajectory_for_inference.shape}"
                            )

                    # Ensure trajectory is in the expected (T, N, D) format for inference
                    if len(trajectory_for_inference.shape) == 2:
                        # If it's (T*N, D), reshape to (T, N, D)
                        T = trajectory.shape[0]
                        N = system.n_particles
                        D = trajectory_for_inference.shape[1]
                        trajectory_for_inference = trajectory_for_inference.reshape(
                            T, N, D
                        )
                        if verbose:
                            console.print(
                                f"[blue]DEBUG:[/blue] Reshaped trajectory to (T, N, D): {trajectory_for_inference.shape}"
                            )

                inferrer = create_inferrer(config.inference)
                model = inferrer.fit(trajectory_for_inference, times)
                progress.update(task4, description="✅ Inference completed")

                results["model"] = model
                results["inferrer"] = inferrer

                if verbose:
                    console.print(
                        f"[green]✓[/green] Dynamics inference completed using {config.inference.method}"
                    )

                if save_intermediate:
                    model_path = output_dir / "model.pkl"
                    inferrer.save_model(model, model_path)
                    if verbose:
                        console.print(f"[blue]💾[/blue] Model saved to {model_path}")

            except Exception as e:
                if verbose:
                    console.print(f"[red]❌[/red] Inference failed: {e}")
                    import traceback

                    console.print(f"[red]❌[/red] Traceback: {traceback.format_exc()}")
                raise e

        # Step 5: Evaluation (optional)
        if config.evaluation is not None and "model" in results:
            task5 = progress.add_task("📊 Evaluating results...", total=None)
            evaluator = create_evaluator(config.evaluation)
            eval_results = evaluator.evaluate(
                model=results["model"],
                trajectory=trajectory,
                times=times,
                system=system,
                output_dir=output_dir,
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
    config_path: str, output_dir: Optional[str] = None, **kwargs
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

    with open(save_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
