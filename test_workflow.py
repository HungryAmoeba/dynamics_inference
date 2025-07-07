#!/usr/bin/env python3
"""Test script to run the workflow and check trajectory."""

import jax.numpy as jnp
from pathlib import Path
from src.dynamo_infer.workflow import run_pipeline_from_config_file


def test_workflow():
    """Test the workflow with a simple configuration."""

    # Create a simple config file
    config_content = """
dynamics:
  type: "swarmalator"
  n_particles: 5
  dimension: 3
  parameters:
    alpha: 2.0
    beta: 4.0
    gamma: 2.0
    J: 0.5
    R: 1.0
    epsilon_a: 0.1
    epsilon_r: 0.1
    noise_strength: 0.0
  initial_conditions:
    position_range: [-2.0, 2.0]

simulation:
  solver: "Tsit5"
  time:
    t0: 0.0
    t1: 2.0
    dt: 0.01
  saveat:
    t0: 0.0
    t1: 2.0
    dt: 0.1
  rtol: 1e-4
  atol: 1e-7

output_dir: "test_output"

inference:
  method: "sindy"
  feature_library:
    type: "polynomial"
    degree: 2
  optimizer: "lasso"
  parameters:
    alpha: 0.1

evaluation:
  save_figures: true
  save_model: true
  save_results: true
"""

    # Write config to file
    config_path = "test_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    try:
        # Run the workflow
        print("Running workflow...")
        results = run_pipeline_from_config_file(config_path, verbose=True)

        # Check results
        print("\n=== Results Analysis ===")
        print(f"Results keys: {list(results.keys())}")

        if "trajectory" in results:
            trajectory = results["trajectory"]
            times = results["times"]

            print(f"Trajectory shape: {trajectory.shape}")
            print(f"Times shape: {times.shape}")
            print(f"Time range: [{times[0]:.3f}, {times[-1]:.3f}]")

            # Check if trajectory is all zeros
            print(f"First state norm: {jnp.linalg.norm(trajectory[0])}")
            print(f"Last state norm: {jnp.linalg.norm(trajectory[-1])}")

            if trajectory.shape[0] > 1:
                print(f"Second state norm: {jnp.linalg.norm(trajectory[1])}")
                print(f"Trajectory[1:] is all zero: {jnp.allclose(trajectory[1:], 0)}")

            # Check for NaN or Inf
            print(f"Trajectory has NaN: {jnp.isnan(trajectory).any()}")
            print(f"Trajectory has Inf: {jnp.isinf(trajectory).any()}")

            # Check state changes
            if trajectory.shape[0] > 1:
                state_changes = jnp.diff(trajectory, axis=0)
                change_norms = jnp.linalg.norm(state_changes, axis=1)
                print(f"Average state change norm: {change_norms.mean():.6f}")
                print(f"Max state change norm: {change_norms.max():.6f}")
                print(f"Min state change norm: {change_norms.min():.6f}")

        if "system" in results:
            system = results["system"]
            print(f"System type: {type(system).__name__}")

            # Test unwrap_state
            if hasattr(system, "unwrap_state"):
                unwrapped = system.unwrap_state(trajectory)
                print(f"Unwrapped state type: {type(unwrapped)}")
                if isinstance(unwrapped, dict):
                    print(f"Unwrapped keys: {list(unwrapped.keys())}")
                    for key, value in unwrapped.items():
                        print(f"  {key} shape: {value.shape}")

        print("\n✅ Workflow completed successfully!")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        if Path(config_path).exists():
            Path(config_path).unlink()


if __name__ == "__main__":
    test_workflow()
