# import itertools
# import jax
# import jax.numpy as jnp
# from jax import grad, jit, random

from scipy.signal import savgol_filter

# from jax.scipy.linalg import solve_triangular
# import numpy as np
# import sympy as sp
# import optax
from jax.experimental.ode import odeint
import sympy as sp
from sympy import Float, Number
from functools import partial
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator
from typing import Optional, List, Dict
from scipy.signal import savgol_filter  # or wherever you import this

# from your_module import BaseFeatureLibrary

from typing import Sequence
from itertools import product


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})


def _zip_like_sequence(x, t):
    """Create an iterable like zip(x, t), but works if t is scalar."""
    if isinstance(t, Sequence):
        return zip(x, t)
    else:
        return product(x, [t])


class GA_DynamicsInference(BaseEstimator):
    def __init__(
        self,
        Gn: int = 3,
        coupling_method: str = "dense",
        coupling_matrix: Optional[jnp.ndarray] = None,
        optimizer: str = "adam",
        feature_library=None,  # type: BaseFeatureLibrary
        differentiation_method: str = "finite_diff",
        feature_names: Optional[List[str]] = None,
        t_default: float = 1.0,
    ):
        # --- 1) Geometry setup ---
        self.Gn = Gn
        if Gn == 3:
            self.g_of_d: List[int] = [0, 1, 1, 1, 2, 2, 2, 3]
        elif Gn == 2:
            self.g_of_d = [0, 1, 1, 2]
        elif Gn == 1:
            self.g_of_d = [0, 1]
        else:
            raise ValueError("Gn must be 1, 2, or 3.")
        # dims for each grade → Python dict of lists
        self.g_to_idxs: Dict[int, List[int]] = {
            g: [i for i, d in enumerate(self.g_of_d) if d == g]
            for g in range(self.Gn + 1)
        }

        # --- 2) Coupling setup ---
        self.coupling_method = coupling_method
        self.coupling_matrix = coupling_matrix
        if coupling_method == "fixed" and coupling_matrix is None:
            raise ValueError("Must provide coupling_matrix for fixed mode.")

        if coupling_method == "gaussian":
            # log‐alpha, log‐eps initial guesses
            self.params = {
                "log_alpha": jnp.log(2.0),
                "log_eps": jnp.log(1.0),
            }
        else:
            self.params = {}

        # --- 3) Opt + feature library + diff method ---
        self.optimizer = optimizer
        self.feature_library = feature_library
        self.differentiation_method = differentiation_method
        self.feature_names = feature_names
        self.t_default = t_default

    def _forward(
        self,
        params: dict,
        dists: jnp.ndarray,  # (T, N, N, G+1)
        feats: jnp.ndarray,  # (T, N, N, M)
        diffs: jnp.ndarray,  # (T, N, N, D)
        K_fixed: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Core dynamics function, previously forward_dyn.
        """
        T, N, N2, D = diffs.shape
        Gp1, M = params["W"].shape

        # 1) build coupling K[t,i,j]
        if self.coupling_method == "fixed":
            K = jnp.broadcast_to(K_fixed, (T, N, N))
        elif self.coupling_method == "learn_fixed":
            K = jnp.broadcast_to(params["K"], (T, N, N))
        elif self.coupling_method == "dense":
            K = jnp.ones((T, N, N))
        else:  # gaussian
            α = jnp.exp(params["log_alpha"])
            ε = jnp.exp(params["log_eps"])
            d1 = dists[..., 1]  # grade‐1 distances
            K = jnp.exp(-(d1**α) / ε)
        # shape check
        assert K.shape == (T, N, N)

        # 2) stack per‐dim weight matrix W_d of shape (M, D)
        Wm_g = params["W"].T  # (M, G+1)
        # static-indexing via Python list → no NonConcreteBooleanIndexError
        W_d = jnp.stack([Wm_g[:, g] for g in self.g_of_d], axis=1)  # (M, D)

        # 3) form F_{t,i,j,d} = sum_m phi[t,i,j,m] * W_d[m,d]
        F_d = jnp.einsum("tijm,md->tijd", feats, W_d)

        # 4) multiply by diffs → R_{t,i,j,d}
        R = F_d * diffs

        # 5) apply coupling → (T, N, D)
        D_out = jnp.einsum("tij,tijd->tid", K, R)
        return D_out

    def fit(self, x: jnp.ndarray, t=None, x_dot=None, u=None, **kwargs):
        """
        Same API as before — just calls self._forward in the loss.
        """
        if t is None:
            t = self.t_default
        # compute x_dot if needed
        if x_dot is None:
            if self.differentiation_method == "savgol":
                x_dot = savgol_filter(
                    x,
                    window_length=25,
                    polyorder=3,
                    axis=0,
                    deriv=1,
                    delta=self.t_default,
                )
                x_dot = jnp.array(x_dot)
            else:
                x_dot = jnp.gradient(x, axis=0) / self.t_default
        # process shapes
        T, N, D = x.shape
        diffs = x[:, :, None, :] - x[:, None, :, :]  # (T,N,N,D)
        # give an option to normalize the diffs along the last axis
        # norm_diff = jnp.linalg.norm(diffs_unnorm, axis = -1) + 1e-8
        # diffs = diffs_unnorm / norm_diff[:, :, :, None]

        # build dists (T,N,N,G+1)
        dists = jnp.stack(
            [
                jnp.linalg.norm(diffs[..., idxs], axis=-1)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        diffs = jnp.concat(
            [
                diffs[..., idxs] * dists[..., g][..., None]
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        # fit features
        flat = dists.reshape((T * N * N, self.Gn + 1))
        self.feature_library.fit(flat)
        feats = self.feature_library.transform(flat)
        M = feats.shape[-1]
        feats = feats.reshape((T, N, N, M))

        # init params["W"] if first time
        key = jax.random.PRNGKey(0)
        if "W" not in self.params:
            self.params["W"] = jax.random.normal(key, (self.Gn + 1, M))

        # choose optimizer
        lr = kwargs.get("lr", 1e-3)
        if self.optimizer.lower() == "adam":
            opt = optax.adam(lr)
        elif self.optimizer.lower() == "adamw":
            opt = optax.adamw(lr, weight_decay=kwargs.get("weight_decay", 1e-4))
        else:
            raise ValueError("Unknown optimizer")

        # loss + step definitions
        def loss_fn(params):
            pred = self._forward(
                params,
                dists,
                feats,
                diffs,
                K_fixed=self.coupling_matrix,
            )
            mse = jnp.mean((pred - x_dot) ** 2)
            reg = kwargs.get("sparsity", 0.0) * jnp.sum(jnp.abs(params["W"]))
            return mse + reg

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # training loop
        params = self.params
        opt_state = opt.init(params)
        epochs = kwargs.get("epochs", int(1e4))
        print_every = kwargs.get("print_every", 1000)
        for e in range(1, epochs + 1):
            params, opt_state, L = step(params, opt_state)
            if e == 1 or e % print_every == 0:
                wmax = jnp.max(jnp.abs(params["W"]))
                print(f"[epoch {e:4d}] loss={L:.3e}, ‖W‖∞={wmax:.3e}")

        self.params = params
        print("Training complete.")

    def predict(self, x: jnp.ndarray, u=None, t=None) -> jnp.ndarray:
        """
        Recompute diffs → dists → feats exactly as in fit, then call _forward.
        """
        if t is None:
            t = self.t_default
        # no x_dot here
        T, N, D = x.shape
        diffs = x[:, :, None, :] - x[:, None, :, :]
        dists = jnp.stack(
            [
                jnp.linalg.norm(diffs[..., idxs], axis=-1)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )
        diffs = jnp.concat(
            [
                diffs[..., idxs] * dists[..., g][..., None]
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        flat = dists.reshape((T * N * N, self.Gn + 1))
        feats = self.feature_library.transform(flat)
        M = self.feature_library.n_output_features_
        feats = feats.reshape((T, N, N, M))

        return self._forward(
            self.params,
            dists,
            feats,
            diffs,
            K_fixed=self.coupling_matrix,
        )

    def _process_trajectories(self, x, t, x_dot):
        if x_dot is None:

            if self.differentiation_method == "savgol":
                # Use Savitzky-Golay filter for smoothing and differentiation

                x_dot = savgol_filter(
                    x,
                    window_length=25,
                    polyorder=3,
                    axis=0,
                    deriv=1,
                    delta=self.t_default,
                )
                x_dot = jnp.array(x_dot)
            else:
                x_dot = jnp.gradient(x, axis=0) / self.t_default

        assert x_dot.shape == x.shape, "Derivatives shape must match time series shape."

        return x, x_dot

    def get_feature_names(self):
        """
        Get the names of the features generated by the feature library.
        Returns
        -------
        list of str
            The names of the features.
        """
        return self.feature_library.get_feature_names()

    def get_simplified_expressions(self, precision=3, significance_threshold=0.05):
        feats = self.feature_library.get_feature_names_symbolic()
        W = jax.device_get(self.params["W"])
        W = jnp.where(jnp.abs(W) < significance_threshold, 0.0, W)
        W = jnp.round(W, decimals=precision)

        exprs = []
        for g in range(self.Gn + 1):
            expr = sum(
                sp.Float(float(W[g, m])) * feats[m]
                for m in range(self.feature_library.n_output_features_)
            )
            expr = sp.simplify(expr)
            exprs.append(round_expr(expr, precision))
        return exprs

    def get_coupling_info(self):
        if self.coupling_method == "fixed":
            return f"Fixed coupling matrix:\n{self.coupling_matrix}\n\n"
        elif self.coupling_method == "gaussian":
            return (
                f"Gaussian coupling with alpha: {jnp.exp(self.params['log_alpha'])}, "
                f"eps: {jnp.exp(self.params['log_eps'])}\n\n"
            )
        return ""

    def print(self, precision=3, **kwargs):
        significance_threshold = kwargs.get("significance_threshold", 0.001)
        exprs = self.get_simplified_expressions(precision, significance_threshold)

        print(self.get_coupling_info())
        print("Learned f_g expressions:")
        for g, expr in enumerate(exprs):
            print(f"f_{g} =")
            sp.pprint(expr, use_unicode=True)
            print()

    def generate_latex_eq(self, output_file, precision=3, **kwargs):
        significance_threshold = kwargs.get("significance_threshold", 0.05)
        exprs = self.get_simplified_expressions(precision, significance_threshold)
        with open(output_file, "w") as f:
            f.write(self.get_coupling_info())
            f.write("Learned $f_g$ expressions:\n\n")

            for g, expr in enumerate(exprs):
                f.write(f"$f_{{{g}}} = {sp.latex(expr)}$\n\n")

    def simulate(self, x0, t, u=None, integrator="odeint", **kwargs):
        """
        Integrate the learned dynamics forward in time.

        Args:
            x0 : jnp.ndarray
                Initial condition, shape (N, D)
            t : jnp.ndarray
                Time points to simulate over, shape (T,)
            u : Ignored for now.
            integrator : str
                Integration method; default is "odeint".
            **kwargs : Extra args for integrator

        Returns:
            Trajectory x(t), shape (T, N, D)
        """

        def dynamics(x_flat, t_scalar):
            # Reshape flattened x to (N, D)
            x = x_flat.reshape(x0.shape)

            # We need to feed a dummy batch with shape (1, N, D) to predict
            x_batch = x[None, :, :]  # shape (1, N, D)

            # Predict dynamics (returns shape (1, N, D))
            dxdt = self.predict(x_batch)[0]  # strip batch dim

            return dxdt.reshape(-1)  # flatten to match odeint interface

        # Initial state flattened
        x0_flat = x0.reshape(-1)

        # Integrate forward
        x_t = odeint(dynamics, x0_flat, t, **kwargs)

        # Reshape result to (T, N, D)
        x_t = x_t.reshape(len(t), *x0.shape)
        return x_t
