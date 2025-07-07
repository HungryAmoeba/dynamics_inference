# Inference Pipeline Implementation

This document describes the complete inference pipeline implementation that has been created based on the existing `gadynamics/inference` codebase.

## Overview

The inference pipeline has been fully restructured and implemented in the new `src/dynamo_infer/inference/` module with a clean, modular architecture that supports the 5-step workflow:

1. **Define configuration** - Type-safe configuration schemas
2. **Simulate dynamics** - Integration with dynamics systems  
3. **Visualize dynamics** - Plotting and visualization
4. **Run inference** - Multiple feature libraries and methods
5. **Evaluate performance** - Comprehensive metrics and figures

## Core Components

### 1. Base Classes (`inference/base.py`)

**BaseInference**: Abstract base class for all inference methods
- `fit(x, t, **kwargs)` - Train the model on trajectory data
- `predict(x, **kwargs)` - Predict derivatives  
- `get_feature_names()` - Get feature descriptions
- `save_model()` / `load_model()` - Model persistence

**DynamicsInferrer**: Extended base class for dynamics-specific inference
- `get_equations(symbolic=True)` - Extract learned equations
- `simulate(x0, t, **kwargs)` - Forward simulation
- `print_equations()` - Pretty-print learned dynamics
- `save_equations()` - Export equations in multiple formats

### 2. Feature Libraries (`inference/feature_library/`)

#### BaseFeatureLibrary (`feature_library/base.py`)
Abstract transformer class following sklearn API:
- `fit(x, y=None)` - Fit library to data
- `transform(x)` - Transform data to features
- `get_feature_names()` - Get feature descriptions
- Support for library composition (`+` for concatenation, `*` for tensor products)

#### MonomialPolynomialLibrary (`feature_library/monomial.py`)
Standard polynomial features using monomial basis:
```python
# Features: 1, x, x^2, x^3, ..., x^degree
library = MonomialPolynomialLibrary(degree=3)
library.fit(data)
features = library.transform(data)  # Shape: (n_samples, n_features * (degree + 1))
```

#### OrthogonalPolynomialLibrary (`feature_library/orthogonal.py`)
Data-adaptive orthogonal polynomials for better numerical conditioning:
```python
# Features: p_0(x), p_1(x), ..., p_degree(x) where p_i are orthonormal
library = OrthogonalPolynomialLibrary(degree=3)
library.fit(data)  # Learns orthogonal basis from data distribution
features = library.transform(data)
```

Key features:
- Uses QR decomposition for numerical stability
- Orthonormal w.r.t. empirical data distribution
- Symbolic expression generation with SymPy
- Support for cross-terms and zero-handling

#### Library Composition
```python
# Concatenate libraries
combined = monomial_lib + orthogonal_lib

# Tensor product of libraries  
tensored = monomial_lib * orthogonal_lib
```

### 3. Inference Models (`inference/models/`)

#### GA_DynamicsInference (`models/ga_inference.py`)
Main geometric algebra dynamics inference model adapted from the original codebase:

**Key Features:**
- **Geometric Algebra**: Supports G1, G2, G3 algebras with grade-based processing
- **Multiple Coupling Methods**: 
  - `"dense"` - All-to-all coupling
  - `"gaussian"` - Distance-based Gaussian coupling  
  - `"fixed"` - User-provided coupling matrix
  - `"learn_fixed"` - Learnable fixed coupling
- **Feature Libraries**: Compatible with all feature libraries
- **Optimization**: Adam/AdamW with JAX
- **Sparsity Regularization**: L1 penalty for feature selection

**Usage:**
```python
config = InferenceConfig(
    method="ga_inference",
    feature_library=FeatureLibraryConfig(
        type="orthogonal_polynomial",
        degree=2
    ),
    coupling_method="gaussian"
)

model = GA_DynamicsInference(config)
model.fit(trajectory_data, times)

# Get learned equations
equations = model.get_equations(symbolic=True)
model.print_equations()

# Simulate forward
predicted_trajectory = model.simulate(x0, future_times)
```

**Architecture:**
- Pairwise difference computation: `diffs = x[:, :, None, :] - x[:, None, :, :]`
- Grade-based distance calculation: `dists[..., g] = norm(diffs[..., grade_g_indices])`
- Feature transformation: `features = library.transform(distances)`
- Coupling application: `dynamics = sum_ij K[i,j] * features[i,j] * diffs[i,j]`

### 4. Utilities (`inference/utils/`)

#### Polynomial Fitting (`utils/polynomial_fits.py`)
Core utilities for orthogonal polynomial computation:

**orth_poly_mc_fast(data_locations, N)**:
- Fast QR-based orthogonal polynomial generation
- Returns both coefficients and evaluations
- Numerically stable for high degrees

**get_symbolic_polynomials(coeffs, varname)**:
- Convert coefficient matrices to SymPy expressions
- Support for pretty-printing and LaTeX export

**evaluate_orthogonal_polynomials(x, coeffs)**:
- Evaluate fitted polynomials at new points
- Efficient vectorized computation

### 5. Evaluation (`evaluation/`)

#### DynamicsEvaluator (`evaluation/base.py`)
Comprehensive evaluation system:

**Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)
- R² coefficient of determination
- Grade-specific metrics (for GA models)

**Visualization:**
- Trajectory comparison plots
- Error over time analysis
- Parity plots (predicted vs true)
- Automatic figure generation and saving

**Usage:**
```python
evaluator = DynamicsEvaluator(config)
results = evaluator.evaluate(
    model=trained_model,
    trajectory=test_data,
    times=test_times,
    system=original_system,  # Optional for ground truth
    output_dir=Path("results/")
)

print(f"MSE: {results['metrics']['mse']:.6f}")
print(f"R²: {results['metrics']['r2']:.6f}")
```

## Factory Functions

The module provides convenient factory functions for creating components:

```python
# Create inferrer from config
inferrer = create_inferrer(config)

# Create feature libraries
monomial_lib = create_monomial_library(degree=3)
orthogonal_lib = create_orthogonal_library(degree=2, include_cross_terms=True)

# Create evaluator
evaluator = create_evaluator(eval_config)
```

## Configuration Integration

The inference system integrates seamlessly with the configuration schema:

```yaml
inference:
  method: "ga_inference"
  feature_library:
    type: "orthogonal_polynomial"
    degree: 2
    parameters:
      include_cross_terms: false
      output_with_zeros: true
  coupling_method: "gaussian"
  differentiation_method: "savgol"
  optimizer: "adam"
  learning_rate: 0.001
  epochs: 10000
  sparsity: 0.01
```

## Key Improvements from Original

1. **Modular Design**: Clean separation of feature libraries, models, and evaluation
2. **Type Safety**: Full type annotations and validation
3. **Extensibility**: Easy to add new feature libraries and inference methods
4. **Documentation**: Comprehensive docstrings and examples
5. **Robustness**: Better error handling and numerical stability
6. **Integration**: Seamless workflow with configuration system
7. **Scientific Output**: Equation export, comprehensive evaluation metrics

## Example Workflow

```python
import jax.numpy as jnp
from dynamo_infer.inference import create_inferrer
from dynamo_infer.evaluation import create_evaluator
from dynamo_infer.config import InferenceConfig, EvaluationConfig

# 1. Configure inference
config = InferenceConfig(
    method="ga_inference",
    feature_library={"type": "orthogonal_polynomial", "degree": 2},
    coupling_method="gaussian"
)

# 2. Create and train model
model = create_inferrer(config)
model.fit(trajectory_data, times)

# 3. Examine learned dynamics
model.print_equations()
equations = model.get_equations(symbolic=True)

# 4. Evaluate performance
evaluator = create_evaluator()
results = evaluator.evaluate(model, test_trajectory, test_times)

# 5. Simulate forward
future_trajectory = model.simulate(x0, future_times)
```

## Migration from Original Codebase

The new implementation maintains API compatibility while providing enhanced functionality:
- **Original**: `GA_DynamicsInference(Gn=3, coupling_method="dense")`
- **New**: `GA_DynamicsInference(config)` with full configuration support

All core functionality from the original `gadynamics/inference` has been preserved and enhanced:
- Geometric algebra processing with grade-based operations
- Multiple coupling methods with learnable parameters
- Feature library flexibility with composition support
- Comprehensive evaluation and equation extraction
- Forward simulation capabilities

The new structure provides a clean foundation for extending with additional inference methods, feature libraries, and evaluation metrics while maintaining the scientific rigor of the original implementation.