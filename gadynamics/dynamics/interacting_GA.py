import jax
import jax.numpy as jnp
import jax.random as jrandom
import sympy as sp

from gadynamics.dynamics.base import DynamicalSystem

from gadynamics.inference.feature_library.monomial_poly_library import (
    MonomialPolynomialLibrary,
)

from typing import List, Dict, Optional

# class InteractingGA(DynamicalSystem):
#     def __init__(self, equation_type = "random"):
#         """
#         Gn: number of nonzero grades (so grades run 0,1,…,Gn)
#         coupling_method: "fixed", "learn_fixed", or "gaussian"
#         coupling_matrix: if fixed, must be an (N,N) array giving K_ij
#         poly_degree: if using our simple PolyFeat, max power
#         """
#         self.Gn = None
#         self.coupling_method = None
#         self.coupling_matrix = None  # shape (N,N)
#         self.equation_type = equation_type
#         if equation_type not in ["random", "random_grade_interaction"]:
#             raise ValueError("equation_type must be 'random' or 'random_grade_interaction'")

#         # will be set by `set_params` and `set_g_of_d`
#         self.params = {}
#         self.g_of_d  = None
#         self.g_to_idxs = {}
#         self.seed = None
#         self.N = None
#         self.poly_degree = None  # max power for polynomial features


#     def initialize(self, config):
#         """
#         Initialize the system with a config dict.
#         Expects keys:
#           - "Gn": number of nonzero grades (so grades run 0,1,…,Gn)
#           - "coupling_method": "fixed", "learn_fixed", or "gaussian"
#           - "coupling_matrix": if fixed, must be an (N,N) array giving K_ij
#           - "poly_degree": if using our simple PolyFeat, max power
#         """
#         Gn = config.get("Gn", self.Gn)
#         self.Gn = Gn if Gn is not None else 3  # default to 3 if not set
#         if Gn == 3:
#             self.g_of_d: List[int] = [0, 1, 1, 1, 2, 2, 2, 3]
#         elif Gn == 2:
#             self.g_of_d = [0, 1, 1, 2]
#         elif Gn == 1:
#             self.g_of_d = [0, 1]
#         else:
#             raise ValueError("Gn must be 1, 2, or 3.")
#         # dims for each grade → Python dict of lists
#         self.g_to_idxs: Dict[int, List[int]] = {
#             g: [i for i, d in enumerate(self.g_of_d) if d == g]
#             for g in range(self.Gn + 1)
#         }
#         self.seed = config.get("seed", 0)

#         self.N = config.get("N", 20)

#         # initialize random function parameters

#         self.coupling_method = config.get("coupling_method", self.coupling_method)
#         self.coupling_matrix = config.get("coupling_matrix", self.coupling_matrix)
#         self.poly_degree = config.get("poly_degree", self.poly_degree)

#         if self.coupling_method == "gaussian":
#             # initialize gaussian kernel parameters
#             self.params = {
#                 "log_alpha": config.get('log_alpha', jnp.log(2) ),
#                 "log_eps": config.get('log_eps', jnp.log(10)),
#                 "scale": config.get('scale', 5.0)
#             }

#         # univariate polynomial functions
#         self.poly_degree = config.get("poly_degree", 2)
#         if self.equation_type == "random":
#             self.feature_library = MonomialPolynomialLibrary(self.poly_degree, include_cross_terms=False)
#             self.feature_library.fit(self.Gn + 1)  # fit to number of grades
#             minval = -5
#             maxval = 5
#             F = jrandom.randint(jrandom.PRNGKey(self.seed), (self.feature_library.n_output_features_, self.Gn + 1), minval, maxval).T
#             # make F sparse now, this p specifies fraction of nonzero entries
#             sparse_p = .3
#             mask = jrandom.bernoulli(jrandom.PRNGKey(self.seed + 1), p=sparse_p, shape=F.shape)
#             F = jnp.where(mask, F, 0.0)  # set some fraction to zero

#         # univariate polynomial functions with interaction terms
#         elif self.equation_type == "random_grade_interaction":
#             self.feature_library = MonomialPolynomialLibrary(self.poly_degree, include_cross_terms=True)
#             self.feature_library.fit(self.Gn + 1) # fit to number of grades
#             minval = -5
#             maxval = 5
#             num_features = (self.poly_degree + 1) ** (self.Gn + 1)
#             F = jrandom.randint(jrandom.PRNGKey(self.seed), (self.feature_library.n_output_features_, self.Gn + 1), minval, maxval).T
#             sparse_p = .2
#             mask = jrandom.bernoulli(jrandom.PRNGKey(self.seed + 1), p=sparse_p, shape=F.shape)
#             F = jnp.where(mask, F, 0.0)  # set some fraction to zero
#         else:
#             raise ValueError("Unknown equation_type: {}".format(self.equation_type))

#         self.params["W"] = F  # W has shape (Gn+1, M) where M is number of features

#         scale = 10
#         random_initial_state = jrandom.normal(jrandom.PRNGKey(self.seed + 4), (self.N, len(self.g_of_d))) * scale

#         # set intiial time to zero
#         random_initial_state = random_initial_state.at[:, 0].set(0.0)  # set grade-0 to zero

#         self.state = random_initial_state  # shape (N, Gn+1)

#     def _forward(
#         self,
#         params: dict,
#         dists: jnp.ndarray,   # (T, N, N, Gn+1)
#         feats: jnp.ndarray,   # (T, N, N, M)
#         diffs: jnp.ndarray,   # (T, N, N, D)
#         K_fixed: jnp.ndarray, # (N, N)
#     ) -> jnp.ndarray:
#         """
#         Core dynamics: D_out[t,i,d] = sum_j K[t,i,j] *
#            [ sum_m feats[t,i,j,m] * W_d[m,d] ] * diffs[t,i,j,d]
#         returns (T, N, D)
#         """
#         T, N, _, D = diffs.shape
#         Gp1, M = params["W"].shape   # W has shape (Gn+1, M)

#         # 1) build K[t,i,j]
#         if self.coupling_method == "fixed":
#             # broadcast your fixed (N,N) matrix to (T,N,N)
#             K = jnp.broadcast_to(K_fixed, (T, N, N))
#         elif self.coupling_method == "learn_fixed":
#             K = jnp.broadcast_to(params["K"], (T, N, N))
#         else:  # gaussian kernel on grade‑1 distances
#             α = jnp.exp(params["log_alpha"])
#             ε = jnp.exp(params["log_eps"])
#             scale = params.get("scale", 1)
#             d1 = dists[..., 1]  # (T,N,N)
#             K  = jnp.exp(-(d1**α) / ε) * scale

#         # sanity check
#         assert K.shape == (T, N, N)
#         # 2) build per‑dim weight matrix W_d of shape (M, D)
#         #    static indexing via Python list avoids NonConcreteBooleanIndexError
#         Wm_g = params["W"].T        # now (M, Gn+1)
#         W_d   = jnp.stack([ Wm_g[:, g] for g in self.g_of_d ], axis=1)  # (M, D)

#         # 3) form F_{t,i,j,d} = sum_m feats[t,i,j,m]*W_d[m,d]
#         #    -> (T,N,N,D)
#         F_d = jnp.einsum("tijm,md->tijd", feats, W_d)

#         # 4) multiply by diffs -> R_{t,i,j,d}
#         R = F_d * diffs

#         # 5) apply coupling and sum_j -> (T,N,D)
#         D_out = jnp.einsum("tij,tijd->tid", K, R)
#         return D_out

#     def compute_derivatives(self, t, state: jnp.ndarray, args) -> jnp.ndarray:
#         # assumes that state is of shape (N, D)
#         if len(state.shape) == 2:
#             state = state[None, :, :]
#         diffs = state[:, None, :, :] - state[:, :, None, :]  # (1, N, N, D)
#         dists = jnp.stack(
#             [
#                 jnp.linalg.norm(diffs[..., idxs], axis=-1)
#                 for g, idxs in sorted(self.g_to_idxs.items())
#             ],
#             axis=-1,
#         )
#         T, N, _, D = diffs.shape
#         flat = diffs.reshape((T * N * N, D))  # (T*N*N, D)

#         feats = self.feature_library.transform(flat)
#         M = self.feature_library.n_output_features_
#         feats = feats.reshape((T, N, N, M))  # (T, N, N, M)

#         deriv = self._forward(
#             self.params,
#             dists,   # (T, N, N, Gn+1)
#             feats,   # (T, N, N, M)
#             diffs,   # (T, N, N, D)
#             K_fixed= self.coupling_matrix  # (N, N)
#         )

#         if state.shape[0] == 1:
#             return deriv[0]
#         else:
#             return deriv

#     def return_state(self):
#         """
#         Return the current state of the system.
#         """
#         return self.state

#     def unwrap_state(self, state: jnp.ndarray):
#         # not really applicable for this model
#         # positions correspond to the grade 1 elements (g_of_d = 1)
#         # orientations correspond to the grade 2 elements (g_of_d = 2)
#         if len(state.shape) == 2:
#             state = state[None, :, :]

#         positions = state[:, :, self.g_to_idxs[1]]
#         orientations = state[:, :, self.g_to_idxs[2]]
#         return positions, orientations

#     def print(self, precision=3, **kwargs):
#         """
#         Print the parameters of the model.
#         """
#         # from sympy import init_printing
#         # init_printing()

#         feats = self.feature_library.get_feature_names_symbolic()
#         W = jax.device_get(self.params["W"])
#         significance_threshold = kwargs.get("significance_threshold", 0.05)
#         W = jnp.where(jnp.abs(W) < significance_threshold, 0.0, W)

#         if self.coupling_method == "fixed":
#             K_fixed = self.coupling_matrix
#             print(f"Fixed coupling matrix: {K_fixed}")
#         elif self.coupling_method == "gaussian":
#             K_fixed = None
#             print(
#                 f"Gaussian coupling with alpha: {jnp.exp(self.params['log_alpha'])}, eps: {jnp.exp(self.params['log_eps'])}"
#             )

#         print(f"Learned f_g expressions")
#         for g in range(self.Gn + 1):
#             _ = [feats[m] for m in range(self.feature_library.n_output_features_)]
#             expr = sum(
#                 [
#                     float(W[g, m]) * feats[m]
#                     for m in range(self.feature_library.n_output_features_)
#                 ]
#             )
#             expr = sp.simplify(expr)
#             print(f"f_{g} = {expr}")
#             sp.pprint(expr)


class InteractingGA(DynamicalSystem):
    def __init__(self, equation_type="random"):
        """
        Gn: number of nonzero grades (so grades run 0,1,…,Gn)
        coupling_method: "fixed", "learn_fixed", or "gaussian"
        coupling_matrix: if fixed, must be an (N,N) array giving K_ij
        poly_degree: if using our simple PolyFeat, max power
        f_type: "poly_feat" (default) or "lorenz_like" (for Gn=3)
        """
        self.Gn = None
        self.coupling_method = None
        self.coupling_matrix = None  # shape (N,N)
        self.equation_type = equation_type
        self.f_type = "poly_feat"  # default, will be set in initialize
        if equation_type not in ["random", "random_grade_interaction"]:
            raise ValueError(
                "equation_type must be 'random' or 'random_grade_interaction'"
            )

        # will be set by `set_params` and `set_g_of_d`
        self.params = {}
        self.g_of_d = None
        self.g_to_idxs = {}
        self.seed = None
        self.N = None
        self.poly_degree = None  # max power for polynomial features

    def initialize(self, config):
        """
        Initialize the system with a config dict.
        Expects keys:
          - "Gn": number of nonzero grades (so grades run 0,1,…,Gn)
          - "coupling_method": "fixed", "learn_fixed", or "gaussian"
          - "coupling_matrix": if fixed, must be an (N,N) array giving K_ij
          - "poly_degree": if using our simple PolyFeat, max power
          - "f_type": "poly_feat" (default) or "lorenz_like"
          - "sigma_init", "rho_init", "beta_init": initial params for lorenz_like
        """
        Gn = config.get("Gn", self.Gn)
        self.Gn = Gn if Gn is not None else 3  # default to 3 if not set
        if Gn == 3:
            self.g_of_d = [0, 1, 1, 1, 2, 2, 2, 3]
        elif Gn == 2:
            self.g_of_d = [0, 1, 1, 2]
        elif Gn == 1:
            self.g_of_d = [0, 1]
        else:
            raise ValueError("Gn must be 1, 2, or 3.")
        # dims for each grade → Python dict of lists
        self.g_to_idxs = {
            g: [i for i, d in enumerate(self.g_of_d) if d == g]
            for g in range(self.Gn + 1)
        }
        self.seed = config.get("seed", 0)
        self.N = config.get("N", 20)

        # Initialize function type
        self.f_type = config.get("f_type", "poly_feat")

        # Initialize coupling parameters
        self.coupling_method = config.get("coupling_method", self.coupling_method)
        self.coupling_matrix = config.get("coupling_matrix", self.coupling_matrix)
        self.poly_degree = config.get("poly_degree", self.poly_degree)

        if self.coupling_method == "gaussian":
            # initialize gaussian kernel parameters
            self.params = {
                "log_alpha": config.get("log_alpha", jnp.log(2)),
                "log_eps": config.get("log_eps", jnp.log(10)),
                "scale": config.get("scale", 5.0),
            }
        else:
            self.params = {}

        # Initialize function-specific parameters
        if self.f_type == "poly_feat":
            # Polynomial feature initialization
            self.poly_degree = config.get("poly_degree", 2)
            if self.equation_type == "random":
                self.feature_library = MonomialPolynomialLibrary(
                    self.poly_degree, include_cross_terms=False
                )
            elif self.equation_type == "random_grade_interaction":
                self.feature_library = MonomialPolynomialLibrary(
                    self.poly_degree, include_cross_terms=True
                )
            else:
                raise ValueError(f"Unknown equation_type: {self.equation_type}")

            self.feature_library.fit(self.Gn + 1)  # fit to number of grades
            minval = -5
            maxval = 5
            num_features = self.feature_library.n_output_features_
            F = jrandom.randint(
                jrandom.PRNGKey(self.seed), (num_features, self.Gn + 1), minval, maxval
            ).T

            sparse_p = 0.3 if self.equation_type == "random" else 0.2
            mask = jrandom.bernoulli(
                jrandom.PRNGKey(self.seed + 1), p=sparse_p, shape=F.shape
            )
            F = jnp.where(mask, F, 0.0)  # sparsify
            self.params["W"] = F  # shape (Gn+1, M)

        elif self.f_type == "lorenz_like":
            # Lorenz-like function initialization
            if self.Gn != 3:
                raise ValueError("lorenz_like f_type requires Gn=3")
            self.params["sigma"] = config.get("sigma_init", 10.0)
            self.params["rho"] = config.get("rho_init", 28.0)
            self.params["beta"] = config.get("beta_init", 8.0 / 3.0)
        else:
            raise ValueError(f"Unknown f_type: {self.f_type}")

        # Initialize particle states
        scale = 10
        random_initial_state = (
            jrandom.normal(jrandom.PRNGKey(self.seed + 4), (self.N, len(self.g_of_d)))
            * scale
        )
        random_initial_state = random_initial_state.at[:, 0].set(0.0)  # grade-0 to zero
        self.state = random_initial_state  # shape (N, D)

    def _forward(
        self,
        params: dict,
        dists: jnp.ndarray,  # (T, N, N, Gn+1)
        feats: jnp.ndarray,  # (T, N, N, M) for poly_feat, else not used
        diffs: jnp.ndarray,  # (T, N, N, D)
        K_fixed: jnp.ndarray,  # (N, N)
    ) -> jnp.ndarray:
        """
        Core dynamics: D_out[t,i,d] = sum_j K[t,i,j] *
           f_g(dists)* (X_i - X_j)_d
        where g = grade of dimension d
        """
        T, N, _, D = diffs.shape

        # 1) Build coupling matrix K[t,i,j]
        if self.coupling_method == "fixed":
            K = jnp.broadcast_to(K_fixed, (T, N, N))  # (T, N, N)
        elif self.coupling_method == "learn_fixed":
            K = jnp.broadcast_to(params["K"], (T, N, N))
        else:  # gaussian kernel on grade-1 distances
            α = jnp.exp(params["log_alpha"])
            ε = jnp.exp(params["log_eps"])
            scale = params.get("scale", 1)
            d1 = dists[..., 1]  # (T,N,N) grade-1 distances
            K = jnp.exp(-(d1**α) / ε) * scale
        assert K.shape == (T, N, N)
        # 2) Compute f_g for each grade and map to dimensions
        if self.f_type == "poly_feat":
            # For poly_feat: F_d = ∑_m feats[t,i,j,m] * W[m, g_of_d[d]]
            M = feats.shape[-1]
            Wm_g = params["W"].T  # (M, Gn+1)
            # Map weights to dimensions: W_d[m,d] = Wm_g[m, g_of_d[d]]
            W_d = jnp.stack([Wm_g[:, g] for g in self.g_of_d], axis=1)  # (M, D)
            F_d = jnp.einsum("tijm,md->tijd", feats, W_d)  # (T,N,N,D)

        elif self.f_type == "lorenz_like":
            # For lorenz_like: compute f_g per grade then map to dimensions
            # dists shape: (T, N, N, 4) [grades 0-3]
            d0, d1, d2, d3 = dists[..., 0], dists[..., 1], dists[..., 2], dists[..., 3]
            sigma, rho, beta = params["sigma"], params["rho"], params["beta"]

            # Compute f_g for each grade (g=0,1,2,3)
            f0 = d0
            f1 = sigma * (d2 - d1)
            f2 = d1 * (rho - d3) - d2
            f3 = d1 * d2 - beta * d3
            F_grade = jnp.stack([f0, f1, f2, f3], axis=-1)  # (T,N,N,4)

            # Map grades to dimensions using g_of_d
            F_d = F_grade[..., self.g_of_d]  # (T,N,N,D)
        else:
            raise ValueError(f"Unsupported f_type: {self.f_type}")

        # 3) Apply interaction: R_{t,i,j,d} = F_d * (X_i - X_j)_d
        R = F_d * diffs  # (T,N,N,D)

        # 4) Sum over j: D_out[t,i,d] = sum_j K[t,i,j] * R[t,i,j,d]
        D_out = jnp.einsum("tij,tijd->tid", K, R)  # (T,N,D)
        return D_out

    def compute_derivatives(self, t, state: jnp.ndarray, args) -> jnp.ndarray:
        # Handle single state or batch
        if len(state.shape) == 2:
            state = state[None, :, :]  # add time dimension
        T = state.shape[0]  # number of time points (usually 1)

        # Compute pairwise differences (T, N, N, D)
        diffs = state[:, None, :, :] - state[:, :, None, :]

        # Compute per-grade distances (T, N, N, Gn+1)
        dists = jnp.stack(
            [
                jnp.linalg.norm(diffs[..., idxs], axis=-1)
                for g, idxs in sorted(self.g_to_idxs.items())
            ],
            axis=-1,
        )
        T, N, _, D = diffs.shape

        # Compute features only for poly_feat
        if self.f_type == "poly_feat":
            flat_diffs = diffs.reshape((T * N * N, -1))  # (T*N*N, D)
            feats = self.feature_library.transform(flat_diffs)  # (T*N*N, M)
            M = feats.shape[-1]
            feats = feats.reshape((T, N, N, M))  # (T, N, N, M)
        else:
            feats = None  # not used for lorenz_like

        # Compute derivatives using core dynamics
        deriv = self._forward(
            self.params, dists, feats, diffs, K_fixed=self.coupling_matrix
        )

        return deriv[0] if state.shape[0] == 1 else deriv

    def return_state(self):
        """
        Return the current state of the system.
        """
        return self.state

    def unwrap_state(self, state: jnp.ndarray):
        # not really applicable for this model
        # positions correspond to the grade 1 elements (g_of_d = 1)
        # orientations correspond to the grade 2 elements (g_of_d = 2)
        if len(state.shape) == 2:
            state = state[None, :, :]

        positions = state[:, :, self.g_to_idxs[1]]
        orientations = state[:, :, self.g_to_idxs[2]]
        return positions, orientations

    def print(self, precision=3, **kwargs):
        """
        Print the parameters of the model.
        """
        # from sympy import init_printing
        # init_printing()

        feats = self.feature_library.get_feature_names_symbolic()
        W = jax.device_get(self.params["W"])
        significance_threshold = kwargs.get("significance_threshold", 0.05)
        W = jnp.where(jnp.abs(W) < significance_threshold, 0.0, W)

        if self.coupling_method == "fixed":
            K_fixed = self.coupling_matrix
            print(f"Fixed coupling matrix: {K_fixed}")
        elif self.coupling_method == "gaussian":
            K_fixed = None
            print(
                f"Gaussian coupling with alpha: {jnp.exp(self.params['log_alpha'])}, eps: {jnp.exp(self.params['log_eps'])}"
            )

        print(f"Learned f_g expressions")
        for g in range(self.Gn + 1):
            _ = [feats[m] for m in range(self.feature_library.n_output_features_)]
            expr = sum(
                [
                    float(W[g, m]) * feats[m]
                    for m in range(self.feature_library.n_output_features_)
                ]
            )
            expr = sp.simplify(expr)
            print(f"f_{g} = {expr}")
            sp.pprint(expr)
