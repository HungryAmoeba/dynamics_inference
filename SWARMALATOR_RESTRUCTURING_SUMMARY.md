# Swarmalator System Restructuring Summary

## Overview

This document summarizes the changes made to restructure the swarmalator system to match the gadynamics implementation and introduce the `promote_to_GA` utility function.

## Key Changes

### 1. Swarmalator System Restructuring (`src/dynamo_infer/dynamics/systems/swarmalator.py`)

**Before:**
- Used 8-dimensional state per particle: `[x1, x2, x3, a1, a2, a3, V, t]`
- Internal state management with volume and time components
- Complex state reshaping between 8D and 2N×D formats

**After:**
- Uses only positions and orientations internally (matches gadynamics)
- State format: `[positions.flatten(), orientations.flatten()]` (shape: `2*N*D`)
- Simplified state management without volume/time components
- Direct compatibility with gadynamics implementation

**Key Changes:**
- Removed 8D state format and volume/time components
- Updated `initialize()` to create state in gadynamics format
- Updated `compute_derivatives()` to work with 2N×D format
- Updated `unwrap_state()` to return only positions and orientations
- Updated `get_expected_state_shape()` to return `(2*N*D,)`

### 2. New `promote_to_GA` Utility Function (`src/dynamo_infer/utils/promote_to_GA.py`)

**Purpose:** Convert unwrapped state data to GA format based on the `Gn` parameter.

**Function Signature:**
```python
def promote_to_GA(Gn: int, unwrapped: Dict[str, jnp.ndarray], times: Optional[jnp.ndarray] = None) -> jnp.ndarray
```

**Behavior by Gn:**
- **Gn = 1**: Output shape `(T, N, 2)`
  - Index 0: Time
  - Index 1: First position component
- **Gn = 2**: Output shape `(T, N, 4)`
  - Index 0: Time
  - Index 1-2: First two position components
  - Index 3: First orientation component
- **Gn = 3**: Output shape `(T, N, 8)`
  - Index 0: Time
  - Index 1-3: All three position components
  - Index 4-6: All three orientation components
  - Index 7: Volume (defaults to zero if not provided)

**Features:**
- Handles missing components gracefully (defaults to zeros)
- Supports both single time arrays and broadcasted time arrays
- Robust error handling for invalid Gn values

### 3. Workflow Integration (`src/dynamo_infer/workflow.py`)

**Updated inference section to use `promote_to_GA`:**
- For GA inference, now uses `promote_to_GA(Gn, unwrapped, times)`
- Automatically detects Gn from config or uses default (3)
- Provides detailed debug output during promotion process
- Maintains backward compatibility with other inference methods

**Key Changes:**
- Replaced manual trajectory reshaping with `promote_to_GA` utility
- Added Gn detection from config
- Improved debug output for troubleshooting

### 4. Utils Module Update (`src/dynamo_infer/utils/__init__.py`)

**Added export for the new utility:**
```python
from .promote_to_GA import promote_to_GA
__all__ = ["promote_to_GA"]
```

## Benefits

### 1. Consistency with Gadynamics
- Internal state format now matches gadynamics exactly
- Same unwrap_state behavior as gadynamics
- Compatible with existing gadynamics workflows

### 2. Simplified State Management
- Removed unnecessary volume and time components from internal state
- Cleaner separation between internal dynamics and external representation
- Easier to understand and maintain

### 3. Flexible GA Promotion
- `promote_to_GA` utility provides flexible conversion to GA format
- Supports different Gn values for different use cases
- Handles missing data gracefully

### 4. Better Workflow Integration
- Seamless integration with inference pipeline
- Automatic Gn detection and promotion
- Improved debugging and error handling

## Testing

All changes have been tested with a comprehensive test suite that verifies:
- Basic swarmalator functionality
- `promote_to_GA` utility for all Gn values
- Trajectory unwrapping and promotion
- Consistency with gadynamics implementation

## Migration Guide

### For Existing Code

**If you were using the 8D state format:**
- Update to use `unwrap_state()` to get positions and orientations
- Use `promote_to_GA()` to convert to GA format for inference

**If you were manually reshaping trajectories:**
- Replace manual reshaping with `promote_to_GA(Gn, unwrapped, times)`
- Let the workflow handle the conversion automatically

**Example Migration:**
```python
# Old way (8D format)
state_8d = state.reshape(N, 8)
positions = state_8d[:, :3]
orientations = state_8d[:, 3:6]

# New way (gadynamics format)
unwrapped = system.unwrap_state(state)
positions = unwrapped['positions']
orientations = unwrapped['orientations']

# For GA inference
ga_traj = promote_to_GA(Gn=3, unwrapped, times)
```

### For New Code

**Basic usage:**
```python
# Create system
system = Swarmalator()
system.initialize(config)

# Get state
state = system.return_state()

# Unwrap for visualization/inference
unwrapped = system.unwrap_state(state)
positions = unwrapped['positions']
orientations = unwrapped['orientations']

# Promote to GA for inference
ga_traj = promote_to_GA(Gn=3, unwrapped, times)
```

## Compatibility

- **Backward Compatible**: The public API remains the same
- **Gadynamics Compatible**: Internal format matches gadynamics exactly
- **Inference Compatible**: Works seamlessly with GA inference pipeline
- **Visualization Compatible**: Unwrapped state works with existing visualizers

## Future Considerations

1. **Volume Support**: If volume dynamics are needed, they can be added as a separate component in the unwrapped state
2. **Time Integration**: Time can be handled externally or added as a separate component
3. **Extension**: The `promote_to_GA` utility can be extended to support additional Gn values or components 