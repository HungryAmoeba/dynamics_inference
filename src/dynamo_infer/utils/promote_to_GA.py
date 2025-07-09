"""Utility function to promote unwrapped state data to GA format."""

import jax.numpy as jnp
from typing import Dict, Optional


def promote_to_GA(
    Gn: int, unwrapped: Dict[str, jnp.ndarray], times: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Promote unwrapped state data to GA format based on Gn.

    Args:
        Gn: Number of nonzero grades (1, 2, or 3)
        unwrapped: Dictionary containing unwrapped state data with keys:
            - 'positions': (T, N, D) array or None
            - 'orientations': (T, N, D) array or None
            - 'volume': (T, N) array or None
        times: Optional time array of shape (T,) or (T, N)

    Returns:
        GA trajectory of shape (T, N, D_GA) where D_GA depends on Gn:
        - Gn=1: D_GA=2 (time, positions)
        - Gn=2: D_GA=4 (time, positions, orientations)
        - Gn=3: D_GA=8 (time, positions, orientations, volume)
    """
    if Gn not in [1, 2, 3]:
        raise ValueError("Gn must be 1, 2, or 3")

    # Get the shapes from the unwrapped data
    positions = unwrapped.get("positions")
    orientations = unwrapped.get("orientations")
    volume = unwrapped.get("volume")

    # Determine the trajectory shape
    if positions is not None:
        T, N, D_pos = positions.shape
    elif orientations is not None:
        T, N, D_ori = orientations.shape
    elif volume is not None:
        T, N = volume.shape
    else:
        raise ValueError(
            "At least one of positions, orientations, or volume must be provided"
        )

    # Initialize output array based on Gn
    if Gn == 1:
        D_GA = 2
    elif Gn == 2:
        D_GA = 4
    elif Gn == 3:
        D_GA = 8
    else:
        raise ValueError("Gn must be 1, 2, or 3")

    # Initialize output array with zeros
    traj = jnp.zeros((T, N, D_GA))

    # Fill in the components based on Gn
    if Gn == 1:
        # Gn=1: [time, positions]
        if times is not None:
            if len(times.shape) == 1:
                times = times[:, None]  # Broadcast to (T, N)
            traj = traj.at[:, :, 0].set(times)

        if positions is not None:
            # Take first dimension of positions
            traj = traj.at[:, :, 1].set(positions[:, :, 0])

    elif Gn == 2:
        # Gn=2: [time, positions, orientations]
        if times is not None:
            if len(times.shape) == 1:
                times = times[:, None]  # Broadcast to (T, N)
            traj = traj.at[:, :, 0].set(times)

        if positions is not None:
            # Take first two dimensions of positions
            for i in range(min(2, positions.shape[-1])):
                traj = traj.at[:, :, 1 + i].set(positions[:, :, i])

        if orientations is not None:
            # Take first dimension of orientations
            traj = traj.at[:, :, 3].set(orientations[:, :, 0])

    elif Gn == 3:
        # Gn=3: [time, positions, orientations, volume]
        if times is not None:
            if len(times.shape) == 1:
                times = times[:, None]  # Broadcast to (T, N)
            traj = traj.at[:, :, 0].set(times)

        if positions is not None:
            # Take first three dimensions of positions
            for i in range(min(3, positions.shape[-1])):
                traj = traj.at[:, :, 1 + i].set(positions[:, :, i])

        if orientations is not None:
            # Take first three dimensions of orientations
            for i in range(min(3, orientations.shape[-1])):
                traj = traj.at[:, :, 4 + i].set(orientations[:, :, i])

        if volume is not None:
            # Set volume in the last dimension
            traj = traj.at[:, :, 7].set(volume)

    return traj
