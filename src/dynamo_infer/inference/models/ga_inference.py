"""Geometric Algebra dynamics inference model."""

from scipy.signal import savgol_filter
from jax.experimental.ode import odeint
import sympy as sp
from sympy import Float, Number
from functools import partial
import jax
import jax.numpy as jnp
import optax
from typing import Optional, List, Dict, Union, Any
from itertools import product

from ..base import DynamicsInferrer
from ...config.schemas import InferenceConfig
from ..feature_library import OrthogonalPolynomialLibrary, MonomialPolynomialLibrary


def round_expr(expr, num_digits):
    """Round numerical expressions to specified digits."""
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Number)})


class GA_DynamicsInference(DynamicsInferrer):
    """
    Geometric Algebra dynamics inference model.

    This model learns dynamics using geometric algebra representations
    and various feature libraries (polynomial, orthogonal, etc.).

    Parameters
    ----------
    config : InferenceConfig
        Configuration for the inference method
    Gn : int, default=3
        Geometric algebra dimension (1, 2, or 3)
    coupling_method : str, default="dense"
        Coupling method ("dense", "gaussian", "fixed", "learn_fixed")
    coupling_matrix : jnp.ndarray, optional
        Fixed coupling matrix (required if coupling_method="fixed")
    feature_library : BaseFeatureLibrary, optional
        Feature library to use. If None, creates based on config
    t_default : float, default=1.0
        Default time step for differentiation
    """

    def __init__(
        self,
        config: InferenceConfig,
        Gn: Optional[int] = None,
        coupling_method: Optional[str] = None,
        coupling_matrix: Optional[jnp.ndarray] = None,
        feature_library=None,
        t_default: float = 1.0,
    ):
        super().__init__(config)

        # Geometry setup - auto-detect Gn if not provided
        if Gn is None:
            # Try to infer Gn from config or use default
            Gn = getattr(config, "Gn", 3)

        self.Gn = Gn
        if Gn == 3:
            self.g_of_d: List[int] = [0, 1, 1, 1, 2, 2, 2, 3]
        elif Gn == 2:
            self.g_of_d = [0, 1, 1, 2]
        elif Gn == 1:
            self.g_of_d = [0, 1]
        else:
            raise ValueError("Gn must be 1, 2, or 3.")

        # Mapping from grades to indices
        self.g_to_idxs: Dict[int, List[int]] = {
            g: [i for i, d in enumerate(self.g_of_d) if d == g]
            for g in range(self.Gn + 1)
        }

        # Coupling setup
        self.coupling_method = coupling_method or config.coupling_method
        self.coupling_matrix = coupling_matrix
        if self.coupling_method == "fixed" and coupling_matrix is None:
            raise ValueError("Must provide coupling_matrix for fixed mode.")

        if self.coupling_method == "gaussian":
            self.params = {
                "log_alpha": jnp.log(2.0),
                "log_eps": jnp.log(1.0),
            }
        else:
            self.params = {}

        # Feature library setup
        if feature_library is None:
            lib_config = config.feature_library
            if lib_config.type == "orthogonal_polynomial":
                self.feature_library = OrthogonalPolynomialLibrary(
                    degree=lib_config.degree, **lib_config.parameters
                )
            elif lib_config.type == "polynomial" or lib_config.type == "monomial":
                self.feature_library = MonomialPolynomialLibrary(
                    degree=lib_config.degree, **lib_config.parameters
                )
            else:
                raise ValueError(f"Unknown feature library type: {lib_config.type}")
        else:
            self.feature_library = feature_library

        # Other parameters
        self.optimizer_name = config.optimizer
        self.differentiation_method = config.differentiation_method
        self.t_default = t_default

        # Training state
        self.fitted = False

    def _forward(
        self,
        params: dict,
        dists: jnp.ndarray,  # (T, N, N, G+1)
        feats: jnp.ndarray,  # (T, N, N, M)
        diffs: jnp.ndarray,  # (T, N, N, D)
        K_fixed: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Core dynamics function.

        Parameters
        ----------
        params : dict
            Model parameters
        dists : jnp.ndarray
            Pairwise distances of shape (T, N, N, G+1)
        feats : jnp.ndarray
            Features of shape (T, N, N, M)
        diffs : jnp.ndarray
            State differences of shape (T, N, N, D)
        K_fixed : jnp.ndarray, optional
            Fixed coupling matrix

        Returns
        -------
        D_out : jnp.ndarray
            Predicted derivatives of shape (T, N, D)
        """
        T, N, N2, D = diffs.shape
        num_grades = max(self.g_of_d) + 1
        Gp1, M = params["W"].shape

        # 1) Build coupling K[t,i,j]
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

        assert K.shape == (T, N, N)

        # 2) Stack per‐dim weight matrix W_d of shape (M, D)
        Wm_g = params["W"].T  # (M, G+1)
        # Static indexing via Python list
        W_d = jnp.stack([Wm_g[:, g] for g in self.g_of_d], axis=1)  # (M, D)

        # 3) Form F_{t,i,j,d} = sum_m phi[t,i,j,m] * W_d[m,d]
        F_d = jnp.einsum("tijm,md->tijd", feats, W_d)

        # 4) Multiply by diffs → R_{t,i,j,d}
        R = F_d * diffs

        # 5) Apply coupling → (T, N, D)
        D_out = jnp.einsum("tij,tijd->tid", K, R)
        return D_out

    def fit(
        self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, **kwargs
    ) -> "GA_DynamicsInference":
        """
        Fit the GA dynamics inference model.

        Parameters
        ----------
        x : jnp.ndarray
            Trajectory data of shape (T, N, D)
        t : jnp.ndarray, optional
            Time points of shape (T,)
        **kwargs : additional parameters
            lr : float, learning rate
            epochs : int, number of epochs
            sparsity : float, sparsity regularization weight
            print_every : int, print frequency

        Returns
        -------
        self : fitted model
        """
        if t is None:
            t = self.t_default

        # Auto-detect Gn based on input dimensions if not already set
        T, N, D = x.shape
        if not hasattr(self, "g_of_d") or self.g_of_d is None:
            # Try to infer the appropriate Gn based on D
            if D == 2:
                self.Gn = 1
                self.g_of_d = [0, 1]
            elif D == 4:
                self.Gn = 2
                self.g_of_d = [0, 1, 1, 2]
            elif D == 8:
                self.Gn = 3
                self.g_of_d = [0, 1, 1, 1, 2, 2, 2, 3]
            else:
                # For non-standard dimensions, create a custom mapping
                # This handles cases like the Swarmalator with 7 dimensions
                print(
                    f"Warning: Non-standard GA dimensions ({D}). Creating custom grade mapping."
                )
                # For 7 dimensions, we'll use a custom mapping
                if D == 7:
                    self.Gn = 2  # Use Gn=2 as base
                    self.g_of_d = [0, 1, 1, 1, 2, 2, 2]  # Custom 7-dim mapping
                else:
                    # Fallback: use sequential mapping
                    self.Gn = min(D - 1, 3)
                    self.g_of_d = list(range(D))

            # Update mapping from grades to indices
            self.g_to_idxs = {
                g: [i for i, d in enumerate(self.g_of_d) if d == g]
                for g in range(max(self.g_of_d) + 1)
            }

        # Compute derivatives if needed
        x_dot = kwargs.get("x_dot", None)
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

        # Process shapes
        T, N, D = x.shape
        diffs = x[:, :, None, :] - x[:, None, :, :]  # (T,N,N,D)

        # Build distance tensors (T,N,N,G+1)
        dists = jnp.stack(
            [
                jnp.linalg.norm(diffs[..., idxs], axis=-1)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        # Normalize differences to unit vectors (with epsilon to avoid division by zero)
        eps = 1e-8
        diffs = jnp.concat(
            [
                diffs[..., idxs] / (dists[..., g][..., None] + eps)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        # Fit feature library
        flat = dists.reshape((T * N * N, len(self.g_to_idxs)))
        self.feature_library.fit(flat)
        feats = self.feature_library.transform(flat)
        M = feats.shape[-1]
        feats = feats.reshape((T, N, N, M))

        # Initialize parameters
        key = jax.random.PRNGKey(kwargs.get("seed", 0))
        if "W" not in self.params:
            # Use the actual number of grades in g_of_d, not just Gn+1
            num_grades = max(self.g_of_d) + 1
            self.params["W"] = jax.random.normal(key, (num_grades, M))

        # Setup optimizer
        lr = kwargs.get("lr", self.config.learning_rate)
        if self.optimizer_name.lower() == "adam":
            opt = optax.adam(lr)
        elif self.optimizer_name.lower() == "adamw":
            opt = optax.adamw(lr, weight_decay=kwargs.get("weight_decay", 1e-4))
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Loss function
        def loss_fn(params):
            pred = self._forward(
                params,
                dists,
                feats,
                diffs,
                K_fixed=self.coupling_matrix,
            )
            mse = jnp.mean((pred - x_dot) ** 2)
            reg = self.config.sparsity * jnp.sum(jnp.abs(params["W"]))
            return mse + reg

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Training loop
        params = self.params
        opt_state = opt.init(params)
        epochs = kwargs.get("epochs", self.config.epochs)
        print_every = kwargs.get("print_every", 1000)

        for e in range(1, epochs + 1):
            params, opt_state, L = step(params, opt_state)
            if e == 1 or e % print_every == 0:
                wmax = jnp.max(jnp.abs(params["W"]))
                print(f"[epoch {e:4d}] loss={L:.3e}, ‖W‖∞={wmax:.3e}")

        self.params = params
        self.fitted = True
        print("Training complete.")

        return self

    def predict(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Predict derivatives using the trained model.

        Parameters
        ----------
        x : jnp.ndarray
            Input trajectory data of shape (T, N, D)

        Returns
        -------
        x_dot_pred : jnp.ndarray
            Predicted derivatives of shape (T, N, D)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        T, N, D = x.shape
        diffs = x[:, :, None, :] - x[:, None, :, :]

        dists = jnp.stack(
            [
                jnp.linalg.norm(diffs[..., idxs], axis=-1)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        # Normalize differences to unit vectors (with epsilon to avoid division by zero)
        eps = 1e-8
        diffs = jnp.concat(
            [
                diffs[..., idxs] / (dists[..., g][..., None] + eps)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )

        flat = dists.reshape((T * N * N, len(self.g_to_idxs)))
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

    def get_feature_names(self) -> List[str]:
        """Get feature names from the feature library."""
        return self.feature_library.get_feature_names()

    def get_equations(self, symbolic: bool = True) -> List[Union[str, sp.Expr]]:
        """
        Get the learned differential equations.

        Parameters
        ----------
        symbolic : bool
            Whether to return symbolic expressions

        Returns
        -------
        equations : list
            Learned equations for each grade
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting equations")

        precision = 3
        significance_threshold = 0.05

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

        if symbolic:
            return exprs
        else:
            return [str(expr) for expr in exprs]

    def simulate(self, x0: jnp.ndarray, t: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Simulate the learned dynamics forward in time.

        Parameters
        ----------
        x0 : jnp.ndarray
            Initial condition of shape (N, D)
        t : jnp.ndarray
            Time points to simulate over of shape (T,)
        **kwargs : additional arguments
            Passed to odeint

        Returns
        -------
        trajectory : jnp.ndarray
            Simulated trajectory of shape (T, N, D)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before simulation")

        def dynamics(x_flat, t_scalar):
            # Reshape flattened x to (N, D)
            x = x_flat.reshape(x0.shape)

            # Feed dummy batch with shape (1, N, D) to predict
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

    def get_coupling_info(self) -> str:
        """Get information about the coupling method."""
        if self.coupling_method == "fixed":
            return f"Fixed coupling matrix:\n{self.coupling_matrix}\n\n"
        elif self.coupling_method == "gaussian":
            return (
                f"Gaussian coupling with alpha: {jnp.exp(self.params['log_alpha'])}, "
                f"eps: {jnp.exp(self.params['log_eps'])}\n\n"
            )
        return f"Coupling method: {self.coupling_method}\n\n"

    def print_equations(self, precision: int = 3) -> None:
        """Print the learned equations."""
        equations = self.get_equations(symbolic=True)

        print(self.get_coupling_info())
        print("Learned f_g expressions:")
        print("=" * 40)
        for g, expr in enumerate(equations):
            print(f"f_{g} = {expr}")
        print("=" * 40)

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        info.update(
            {
                "geometric_algebra_dim": self.Gn,
                "coupling_method": self.coupling_method,
                "feature_library_type": type(self.feature_library).__name__,
                "n_features": getattr(self.feature_library, "n_output_features_", None),
                "optimizer": self.optimizer_name,
            }
        )
        return info
