#!/usr/bin/env python3
"""
Test script to verify chiral metamaterial integration with dynamo_infer pipeline.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dynamo_infer.config import load_config
from src.dynamo_infer.workflow import run_full_pipeline
from src.dynamo_infer.dynamics import create_system
from src.dynamo_infer.config.schemas import DynamicsConfig
from src.dynamo_infer.dynamics.systems.chiral_metamaterial import ChiralMetamaterial


def test_system_creation():
    """Test creating the chiral metamaterial system directly."""
    print("Testing chiral metamaterial system creation...")
    
    config = DynamicsConfig(
        type="chiral_metamaterial",
        n_particles=0,  # Will be set by the system
        dimension=3,
        parameters={
            "L": 1.0, 
            "k_e": 1e4, 
            "r": 0.2, 
            "k_r": 1e6, 
            "gamma": 1.0, 
            "gamma_t": 0.1, 
            "dt": 1e-3,
            "grid_type": "3d", 
            "grid_size": [4, 4, 4],  # Smaller grid for testing
            "forcing": {
                "type": "velocity_constraint", 
                "v_top_z": -0.25, 
                "t_release": 5.0
            },
            "seed": 42
        },
        initial_conditions={
            "position_noise": 0.01, 
            "angle_noise": 0.01, 
            "velocity_noise": 0.01
        }
    )
    
    # Create and initialize system
    system = create_system(config)
    print(f"✓ System created: {type(system).__name__}")
    print(f"✓ Number of particles: {system.n_particles}")
    print(f"✓ Dimension: {system.dimension}")
    print(f"✓ State shape: {system.get_expected_state_shape()}")
    
    # Check if it's a ChiralMetamaterial to access specific methods
    if isinstance(system, ChiralMetamaterial):
        print(f"✓ Graph info: {system.get_graph_info()}")
    else:
        print(f"✓ System type doesn't have graph info")
    
    # Test state unwrapping
    state = system.return_state()
    unwrapped = system.unwrap_state(state)
    print(f"✓ Unwrapped state components: {list(unwrapped.keys())}")
    print(f"✓ Positions shape: {unwrapped['positions'].shape}")
    print(f"✓ Angles shape: {unwrapped['angles'].shape}")
    
    # Test derivatives computation
    derivatives = system.compute_derivatives(0.0, state)
    print(f"✓ Derivatives computed, shape: {derivatives.shape}")
    
    return system


def test_config_file():
    """Test loading from configuration file."""
    print("\nTesting configuration file loading...")
    
    try:
        config = load_config("configs/chiral_metamaterial.yaml")
        print(f"✓ Config loaded: {config.dynamics.type}")
        
        # Create system from config
        system = create_system(config.dynamics)
        print(f"✓ System created from config: {type(system).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Config file test failed: {e}")
        return False


def test_pipeline_run():
    """Test running a minimal pipeline."""
    print("\nTesting minimal pipeline run...")
    
    # Create a minimal config for fast testing
    config_dict = {
        "output_dir": "outputs/test_chiral",
        "dynamics": {
            "type": "chiral_metamaterial",
            "n_particles": 0,
            "dimension": 3,
            "parameters": {
                "L": 1.0, "k_e": 1e4, "r": 0.2, "k_r": 1e6, 
                "gamma": 1.0, "gamma_t": 0.1, "dt": 1e-3,
                "grid_type": "3d", "grid_size": [3, 3, 3],  # Very small grid
                "forcing": {"type": "velocity_constraint", "v_top_z": -0.25, "t_release": 2.0},
                "seed": 42
            },
            "initial_conditions": {
                "position_noise": 0.01, "angle_noise": 0.01, "velocity_noise": 0.01
            }
        },
        "simulation": {
            "solver": "Tsit5",
            "time": {"t0": 0.0, "t1": 1.0, "dt": 0.1},  # Short simulation
            "saveat": {"t0": 0.0, "t1": 1.0, "dt": 0.1},
            "rtol": 1e-4, "atol": 1e-7
        }
    }
    
    try:
        # Create config from dict
        from src.dynamo_infer.config.core import Config
        config = Config.from_dict(config_dict)
        
        # Run pipeline (simulation only, no visualization/inference)
        results = run_full_pipeline(
            config, 
            verbose=True, 
            save_intermediate=False
        )
        
        print(f"✓ Pipeline completed successfully")
        print(f"✓ Trajectory shape: {results['trajectory'].shape}")
        print(f"✓ Time points: {len(results['times'])}")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Chiral Metamaterial Integration with dynamo_infer")
    print("=" * 60)
    
    # Test 1: System creation
    try:
        system = test_system_creation()
        test1_passed = True
    except Exception as e:
        print(f"✗ System creation test failed: {e}")
        import traceback
        traceback.print_exc()
        test1_passed = False
    
    # Test 2: Config file loading
    test2_passed = test_config_file()
    
    # Test 3: Pipeline run
    test3_passed = test_pipeline_run()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"System Creation:     {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Config File Loading: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Pipeline Run:        {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall Result:      {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Chiral metamaterial system is successfully integrated with dynamo_infer!")
        print("You can now use it in the pipeline with:")
        print("  python -m src.dynamo_infer.workflow configs/chiral_metamaterial.yaml")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)