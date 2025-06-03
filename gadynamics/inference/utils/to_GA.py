import jax.numpy as jnp
import jax
import numpy as np


def pos_ori_to_GA(pos, ori, t=None, vol=None):
    """
    Takes in position and orientation data, either in 2D or 3D and returns
    the corresponding trajectory in GA

    """
    if len(pos.shape) != 3 or len(ori.shape) != 3:
        raise ValueError("Expected input to be of shape [T, N, D]")
    if t is not None:
        assert len(t) == len(pos)
    if vol is not None:
        assert len(vol) == len(pos)

    T, N, D = pos.shape

    if D not in {2, 3}:
        raise ValueError("Only supported for ")

    if D == 2 and vol is not None:
        vol = None
        print("Warning: volumne unused. Be careful about the mapping")

    traj = jnp.zeros((T, N, 2**D))
    if t is not None:
        if len(t.shape) == 1:
            # broadcast t to shape (T, N)
            t = t[:, None]
        traj = traj.at[:, :, 0].set(t)
    if vol is not None:
        if len(vol.shape) == 1:
            # broadcast vol to shape (T, N)
            vol = vol[:, None]
        traj = traj.at[:, :, -1].set(vol.squeeze(-1))

    traj = traj.at[:, :, 1 : D + 1].set(pos)
    if D == 2:
        traj = traj.at[:, :, -1].set(ori)
    if D == 3:
        traj = traj.at[:, :, D + 1 : -1].set(ori)

    return traj
