import itertools
import jax
import jax.numpy as jnp
from jax import grad, jit, random

from scipy.signal import savgol_filter

from jax.scipy.linalg import solve_triangular
import numpy as np
import sympy as sp
import optax

from .base import BaseInference
from .utils import polynomial_fits as poly


def compute_tensor_basis_eval(univ_eval):
    """
    Build the full tensor‐product basis evaluations φ of shape
      (T, N, N, B**G)
    from a list of G univariate eval arrays.

    Parameters
    ----------
    univ_eval : list of length G
        Each element is an array of shape (T, N, N, B), the evaluations
        of the grade‐g univariate polynomial basis at each (t,i,j).

    Returns
    -------
    phi : jnp.ndarray, shape (T, N, N, B**G)
        φ[t,i,j,p] is the product over grades g of
        univ_eval[g][t,i,j,m_g],
        where p is the flattened multi‐index (m_0,…,m_{G-1}).
    """
    G = len(univ_eval)
    letters = [chr(ord("A") + i) for i in range(G)]  # ['A','B','C',...]
    in_subs = ",".join(f"tij{L}" for L in letters)  # "tijA,tijB,..."
    out_subs = "tij" + "".join(letters)  # "tijABC..."
    subs = f"{in_subs}->{out_subs}"

    # pack into one big einsum:
    stacked = jnp.einsum(subs, *univ_eval)
    T, N, _, *tail = stacked.shape
    B = univ_eval[0].shape[-1]
    return stacked.reshape(T, N, N, B**G)


def evaluate_polynomials_at(pts, coeffs):
    """
    pts     : array‐like, shape (L,)
              new x‐locations where you want p_k(x)
    coeffs  : ndarray, (B, B)   # coeffs[j, k] = coeff of x^j in p_k

    Returns
    -------
    P_new   : ndarray, (L, B)
              P_new[i, k] = p_k( pts[i] ).
    """
    pts = jnp.asarray(pts)
    B = coeffs.shape[0]
    # 1) Vandermonde at the L new points
    V = jnp.vander(pts, B, increasing=True)  # (L, B)

    # 2) multiply to get all B polynomials at once
    P_new = V @ coeffs  # (L, B)
    return P_new


def evaluate_polynomial_basis(dists, coeffs):
    """
    Evaluate the polynomial basis at a given point x with coefficients coeffs.

    Parameters
    ----------
    dists : list[jnp.ndarray], len (G), each shape (T, N, N)
        The input data points.
    coeffs : list[jnp.ndarray], len(G), each shape (B, B)
        The coefficients of the polynomial basis.

    Returns
    -------
    result : list of jnp.ndarray, each shape (T, N, N, B)
    """
    # Evaluate the polynomial basis at each point in x to get the univariate evaluations
    univ_eval = []  # want this to be of length G, each elt (T,N,N,B)

    for g in range(len(coeffs)):
        # Get the coefficients for the g-th polynomial basis
        Cg = coeffs[g]
        # Get the distances for the g-th polynomial basis
        dists_g = dists[..., g]  # shape (T, N, N)
        # Evaluate the polynomial basis at each point in dists_g
        P_eval = evaluate_polynomials_at(dists_g.ravel(), Cg)  # shape (T*N*N, B)
        P_eval = P_eval.reshape(dists_g.shape + (-1,))  # shape (T, N, N, B)
        # Append the evaluated polynomial basis to the list
        univ_eval.append(P_eval)

    return univ_eval


def evaluate_tensor_basis(dists, coeffs):
    """
    Evaluate the tensor‐product basis at a given point x with coefficients coeffs.

    Parameters
    ----------
    dists : list[jnp.ndarray], len (G), each shape (T, N, N)
        The input data points.
    coeffs : list[jnp.ndarray], len(G), each shape (B, B)
        The coefficients of the polynomial basis.

    Returns
    -------
    result : jnp.ndarray, shape (T, N, N, M)
        The evaluated tensor‐product basis, where M = B**G.
    """
    univ_eval = evaluate_polynomial_basis(dists, coeffs)

    return compute_tensor_basis_eval(univ_eval)


def forward_dyn(
    params,
    X,
    dists,
    univ_eval,
    coupling_mode,
    g_of_d,
    K_fixed=None,
    ext_derivative_fxn=None,
):
    """
    params:        dict with
                    'W':    shape (G, M)   ← note the grade‐major layout
                    'K' or ('log_alpha','log_eps')
    X:             (T, N, D)
    dists:         (T, N, N, 4)
    univ_eval:     list of 4 arrays, each (T, N, N, B)
    coupling_mode: 'fixed' or 'gaussian'
    returns        D_out of shape (T, N, D)
    """
    T, N, D = X.shape
    # 0) assemble the full tensor‐product basis  (T,N,N,M)
    phi = compute_tensor_basis_eval(univ_eval)
    # 1) build coupling K_{t,i,j}
    if coupling_mode == "fixed":
        K = jnp.broadcast_to(K_fixed, (T, N, N))
    elif coupling_mode == "learn_fixed":
        K = jnp.broadcast_to(params["K"], (T, N, N))
    else:
        α = jnp.exp(params["log_alpha"])
        ε = jnp.exp(params["log_eps"])
        d1 = dists[..., 1]  # grade‐1 distances
        K = jnp.exp(-(d1**α) / ε)

    if params.get("individual_terms", None) is not None:
        individual_terms_matrix = params["individual_terms"]

    # 2) compute pairwise diffs Δ_{t,i,j,d}
    diffs = X[:, :, None, :] - X[:, None, :, :]  # (T,N,N,D)

    # 4) pull W into (M,G) then select per‐dim weights → W_d (M,D)
    W = params["W"]  # (G, M)
    Wm_g = W.T  # now (M, G)
    W_d = Wm_g[:, g_of_d]  # (M, D)

    # 5) form F_{t,i,j,d} = sum_m phi[t,i,j,m] * W_d[m,d]  → (T,N,N,D)
    #    einsum indices:  m sums, d stays
    F_d = jnp.einsum("tijm,md->tijd", phi, W_d)

    # 6) multiply by diffs → R_{tij,d}
    R = F_d * diffs  # (T,N,N,D)

    # 7) apply coupling: sum_j K[t,i,j] * R[t,i,j,d]  → (T,N,D)
    D_out = jnp.einsum("tij,tijd->tid", K, R)

    # 8) apply external derivative function if provided
    if ext_derivative_fxn is not None:
        D_out = D_out + ext_derivative_fxn(D_out, X)

    if params.get("individual_terms", None) is not None:
        # individual_terms_matrix has shape [D*B, D]
        DB, _ = individual_terms_matrix.shape
        B = DB // D
        # make the vandermonde matrix of X (T, N, D)
        V = jnp.vander(X.reshape(-1), B, increasing=True)  # shape (T*N*D, B)
        V = V.reshape((T, N, D * B))  # shape (T, N, D, B)
        # now we need to multiply the Vandermonde matrix by the individual terms
        individ_out = jnp.einsum(
            "tnh,hd->tnd", V, individual_terms_matrix
        )  # shape (T, N, D)
        D_out += individ_out

    return D_out


class infer_dynamics(BaseInference):
    def __init__(
        self,
        time_series,
        g_of_d,
        derivatives=None,
        sparsity_alpha=0.0,
        coupling=None,
        coupling_mode="fixed",
        max_poly_degree=3,
        covariant=True,
        ext_derivative_fxn=None,
        learned_individual_terms=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.X = jnp.asarray(time_series)  # shape (T,N,D)
        if type(derivatives) is str or derivatives is None:
            self.derivatives = derivatives
        else:
            self.derivatives = jnp.asarray(derivatives)
        # self.derivatives = (
        #     jnp.asarray(derivatives) if type(derivatives) is not str or derivatives is not None else derivatives
        # )
        self.N, self.T, self.D = (
            time_series.shape[1],
            time_series.shape[0],
            time_series.shape[2],
        )
        self.B = max_poly_degree + 1  # # univariate basis functions
        self.coupling_mode = coupling_mode
        self.sparsity_alpha = sparsity_alpha
        self.g_of_d = g_of_d  # maybe should make this G_n
        assert self.g_of_d.shape == (self.D,), "g_of_d must be a 1D array of length D"
        self.G = self.g_of_d.max() + 1  # number of grades
        self.M = self.B**self.G  # total tensor‐product basis size
        self.ext_derivative_fxn = ext_derivative_fxn
        self.learned_individual_terms = learned_individual_terms

        diffs = self.X[:, :, None, :] - self.X[:, None, :, :]
        dists_list = []
        for g in range(self.G):
            # find which output‐dims belong to grade g
            idxs = jnp.nonzero(self.g_of_d == g)[0]
            # take those channels out of diffs and norm over last axis
            #   -> shape (T,N,N)
            dists_g = jnp.linalg.norm(diffs[..., idxs], axis=-1)
            dists_list.append(dists_g)
        # stack into (T,N,N,G)

        self.dists = jnp.stack(dists_list, axis=-1)

        # 3) for each grade, fit univariate orthonormal polys
        self.univ_coefs = []  # list of G arrays, each (B,B)
        self.univ_eval = []  # list of G arrays, each (T,N,N,B)
        K = self.T * self.N * self.N  # total points per grade
        for g in range(self.G):
            # flatten the grade‐g distances
            flat = self.dists[..., g].ravel()  # reshape((K,))   # shape (K,)
            # returns coeffs (B×B) and evaluations (K×B)
            Cg, P_eval_flat = poly.orth_poly_mc_fast(flat, max_poly_degree)
            # reshape evals back to (T,N,N,B)
            P_eval = P_eval_flat.reshape((self.T, self.N, self.N, self.B))

            self.univ_coefs.append(Cg)
            self.univ_eval.append(P_eval)

        # 3) Randomly initialize all learnable params
        key = random.PRNGKey(0)
        k1, k2 = random.split(key)
        k3, k4 = random.split(k2)
        # 3a) Tensor‐product polynomial weights W[d,k]:
        #     D output dimensions × M basis functions
        self.params = {}
        self.params["W"] = random.normal(k1, (self.G, self.M))
        self.fixed_K = None
        if coupling is not None and coupling_mode == "fixed":
            self.fixed_K = coupling
        # 3b) Coupling parameters:
        elif coupling_mode == "learn_fixed" and coupling is None:
            # full NxN matrix (we’ll learn it directly)
            self.params["K"] = 1e-2 * random.normal(k2, (self.N, self.N))
        elif coupling_mode == "gaussian":
            # exponent alpha and scale epsilon (rescaled to be positive via softplus)
            self.params["log_alpha"] = jnp.log(2.0)
            self.params["log_eps"] = jnp.log(1.0)
        else:
            raise ValueError("mode must be 'fixed', 'gaussian', or 'learn_fixed'")

        if self.learned_individual_terms:
            # create some parameters which are going to covariant terms. but not yet lol
            self.params["individual_terms"] = random.normal(
                k3, (self.D * self.B, self.D)
            )

    def preprocess_data(self, data):
        """
        Preprocess the input time series data and compute derivatives if not provided.

        Parameters
        ----------
        data : jnp.ndarray
            The input time series data with shape (T, N, D).

        Returns
        -------
        processed_data : jnp.array
            The preprocessed time series data.
        targets : jnp.array
            The target values for the data.
        """

        if self.derivatives is None or type(self.derivatives) is str:
            # Compute numerical derivatives if not provide
            dt_est = (data[-1, 0, 0] - data[0, 0, 0]) / data.shape[
                0
            ]  # Assuming uniform time steps
            if self.derivatives == "savgol":
                # Use Savitzky-Golay filter for smoothing and differentiation

                self.derivatives = savgol_filter(
                    data, window_length=25, polyorder=3, axis=0, deriv=1, delta=dt_est
                )
                self.derivatives = jnp.array(self.derivatives)
            else:
                self.derivatives = jnp.gradient(data, axis=0) / dt_est

        assert (
            self.derivatives.shape == data.shape
        ), "Derivatives shape must match time series shape."

        return data, self.derivatives

    def fit(
        self,
        lr: float = 1e-3,
        epochs: int = 200,
        optimizer_name: str = "adam",
        print_every: int = 100,
    ):
        # 0) make sure derivatives exist
        if self.derivatives is None:
            self.X, self.derivatives = self.preprocess_data(self.X)

        # 1) pull locals
        X = self.X
        dists = self.dists
        deriv = self.derivatives
        univ_eval = tuple(self.univ_eval)
        coupling_mode = self.coupling_mode
        g_of_d = self.g_of_d
        sparsity = self.sparsity_alpha
        params = self.params
        K_fixed = self.fixed_K
        ext_derivative_fxn = self.ext_derivative_fxn

        # 2) pick an Optax optimizer
        if optimizer_name.lower() == "adam":
            optimizer = optax.adam(lr)
        elif optimizer_name.lower() == "adamw":
            optimizer = optax.adamw(lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # 3) create optimizer state
        opt_state = optimizer.init(params)

        # 4) define your loss outside the loop
        def loss_fn(
            params,
            X,
            dists,
            univ_eval,
            deriv,
            sparsity,
            K_fixed=None,
            ext_derivative_fxn=None,
        ):
            pred = forward_dyn(
                params,
                X,
                dists,
                univ_eval,
                coupling_mode,
                g_of_d,
                K_fixed=K_fixed,
                ext_derivative_fxn=ext_derivative_fxn,
            )
            mse = jnp.mean((pred - deriv) ** 2)
            reg = sparsity * jnp.sum(jnp.abs(params["W"]))
            if params.get("individual_terms", None) is not None:
                # add the regularization term for the individual terms
                reg += sparsity * jnp.sum(jnp.abs(params["individual_terms"]))
            return mse + reg

        # 5) combine loss+grad+update into one jitted step
        @jax.jit
        def step(params, opt_state, X, dists, univ_eval, deriv, sparsity):
            loss, grads = jax.value_and_grad(loss_fn)(
                params,
                X,
                dists,
                univ_eval,
                deriv,
                sparsity,
                K_fixed,
                ext_derivative_fxn,
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # 6) training loop
        for epoch in range(1, epochs + 1):
            params, opt_state, loss = step(
                params, opt_state, X, dists, univ_eval, deriv, sparsity
            )

            if epoch % print_every == 0 or epoch == 1:
                # example diagnostics
                wmax = jnp.max(jnp.abs(params["W"]))
                print(f"[epoch {epoch:3d}] loss={loss:.3e}, ‖W‖∞={wmax:.3e}")

        # 7) save back
        self.params = params
        print("Training complete.")
        # return a final prediction
        pred = forward_dyn(
            params,
            X,
            dists,
            univ_eval,
            coupling_mode,
            g_of_d,
            K_fixed=K_fixed,
            ext_derivative_fxn=ext_derivative_fxn,
        )
        return pred

    def predict(self, X):
        """
        Predict outcomes using the trained model.

        Parameters
        ----------
        X : jnp.ndarray
            The preprocessed time series data.

        Returns
        -------
        predictions : jnp.ndarray
            The predicted derivatives of the time series data.
        """
        # Here we would implement the prediction logic based on the fitted model
        # For now, we will return a placeholder

        if self.params is None:
            raise ValueError(
                "Model parameters are not set. Please fit the model first."
            )
        params = self.params

        # Ensure X is a JAX array
        X = jnp.asarray(X)  # Ensure X is a JAX array

        # Recompute distances and univariate evaluations
        T, N, D = X.shape

        diffs = X[:, :, None, :] - X[:, None, :, :]  # (T, N, N, D)
        dists = []
        g_of_d = self.g_of_d
        for g in range(self.G):
            idxs = jnp.nonzero(g_of_d == g)[0]
            dists_g = jnp.linalg.norm(diffs[..., idxs], axis=-1)
            dists.append(dists_g)
        dists = jnp.stack(dists, axis=-1)  # shape (T, N, N, G)
        # 1) build univariate evaluations
        univ_eval = evaluate_polynomial_basis(dists, self.univ_coefs)

        mode = self.coupling_mode
        K = self.fixed_K

        import pdb

        pdb.set_trace()
        pred = forward_dyn(
            params,
            X,
            dists,
            univ_eval,
            mode,
            g_of_d,
            K_fixed=K,
            ext_derivative_Fxn=self.ext_derivative_fxn,
        )

        return pred

    def print_equation(self, threshold=1e-3, max_terms=20):
        """
        Print a human‐readable summary of the inferred dynamics:
         - coupling (fixed K or Gaussian params)
         - univariate bases p_g^i(d_g)
         - top tensor‐product terms in each dimension
        """
        B = self.B  # num basis per grade
        M = self.M  # total tensor‐product size = B**4
        D = self.D  # output dimensions

        # 1) coupling
        print("=== Coupling ===")
        if self.coupling_mode == "fixed":
            K = np.array(self.fixed_K)
            print("Fixed coupling matrix K (shape: N×N):")
            print(K)
        else:
            α = float(jnp.exp(self.params["log_alpha"]))
            ε = float(jnp.exp(self.params["log_eps"]))
            print(f"Gaussian coupling A(d₁) = exp(- d₁^{α:.3f} / {ε:.3f})")

        # 2) univariate basis
        print("\n=== Univariate bases p_g^i(d_g) ===")
        for g in range(self.G):
            Cg = np.array(self.univ_coefs[g])  # shape (B, B)
            print(f"\n grade {g}  (d₍grade₎ → basis functions)")
            for i in range(B):
                coeffs = Cg[:, i]
                # Replace NaN or Inf values with 0
                coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
                # build monomial string
                terms = []
                for j, c in enumerate(coeffs):
                    if abs(c) < threshold:
                        continue
                    if j == 0:
                        terms.append(f"{c:.3f}")
                    else:
                        terms.append(f"{c:.3f}·d{g}^{j}")
                poly_str = " + ".join(terms) if terms else "0"
                print(f"  p_{g}^{i}(d{g}) = {poly_str}")

        # 3) tensor‐product expansions
        print("\n=== Tensor‐product terms f_d(d₀,d₁,d₂,d₃) ===")
        W = np.array(self.params["W"])  # shape (D, M)
        G, _ = W.shape
        for d in range(G):
            print(f"\n— grade {d} —")
            row = W[d]
            # pick top |row| entries
            idxs = np.argsort(-np.abs(row))[:max_terms]
            for idx in idxs:
                w = row[idx]
                if abs(w) < threshold:
                    continue
                # decode multi‐index in base‑B
                i0 = idx % B
                i1 = (idx // B) % B
                i2 = (idx // (B**2)) % B
                i3 = (idx // (B**3)) % B
                print(
                    f"  {w:.3e} · p₀^{i0}(d₀) · p₁^{i1}(d₁) · p₂^{i2}(d₂) · p₃^{i3}(d₃)"
                )

        # 4) generic dynamics formula
        print("\n=== Final inferred dynamics ===")
        if self.coupling_mode == "fixed":
            print("For each index α:")
            print("  dX_α/dt = ∑_γ K[α,γ] · f(d₀,d₁,d₂,d₃) · (X_α – X_γ)")
        else:
            print("For each index α:")
            print("  dX_α/dt = ∑_γ exp(-d₁_{αγ}^α / ε) · f(d₀,d₁,d₂,d₃) · (X_α – X_γ)")
        print("where f(d₀,d₁,d₂,d₃) is the polynomial above.")

    def get_symbolic_internal(self, var_names, coeff_threshold=1e-6):
        """
        Build a list of Sympy expressions for the “individual terms” part:
            ∑_{d_in=0}^{D-1} ∑_{b=0}^{B-1} c_{d_in,b,d_out} * var[d_in]**b
        Only keeps coefficients |c| > coeff_threshold.
        """
        assert "individual_terms" in self.params, "No individual_terms learned."
        B = self.B
        D = self.D

        # sympy symbols for your variables, e.g. ['e0','x1','x2',…]
        syms = sp.symbols(var_names, real=True)

        # pull numpy array for ease of indexing
        C = np.array(self.params["individual_terms"])  # shape (D*B, D)

        internal_exprs = []
        for d_out in range(D):
            expr = 0
            for d_in, sym in enumerate(syms):
                for b in range(B):
                    c = float(C[d_in * B + b, d_out])
                    if abs(c) > coeff_threshold:
                        expr += c * sym**b
            internal_exprs.append(sp.simplify(expr))

        return internal_exprs

    def get_symbolic_tensor_basis(self):
        """
        Reconstruct all M = B**G tensor‐product basis functions φ_m
        as Sympy expressions in the graded distances d_0, d_1, …, d_{G-1}.
        """
        G = self.G
        B = self.B
        M = self.M

        # first build each grade’s univariate polys p_{g,k}(d_g) = ∑_j Cg[j,k] d_g**j
        polys = []
        for g in range(G):
            Cg = np.array(self.univ_coefs[g])  # (B, B)
            dg = sp.symbols(f"d{g}", real=True)
            pols_g = []
            for k in range(B):
                p = sum(float(Cg[j, k]) * dg**j for j in range(B))
                pols_g.append(sp.simplify(p))
            polys.append(pols_g)

        # now form the M tensor‐product combinations
        tensor_exprs = []
        for m in range(M):
            mm = m
            factors = []
            for g in range(G):
                m_g = mm % B
                mm //= B
                factors.append(polys[g][m_g])
            tensor_exprs.append(sp.simplify(sp.prod(factors)))

        return tensor_exprs

    def get_symbolic_coupling(self):
        """
        Return a Sympy expression or Matrix for your coupling K.
        """
        if self.coupling_mode == "gaussian":
            α = float(np.exp(self.params["log_alpha"]))
            ε = float(np.exp(self.params["log_eps"]))
            d1 = sp.symbols("d1", real=True)
            return sp.exp(-(d1**α) / ε)

        elif self.coupling_mode == "learn_fixed":
            K = np.array(self.params["K"])
            return sp.Matrix(K)

        elif self.coupling_mode == "fixed":
            K = np.array(self.fixed_K)
            return sp.Matrix(K)

        else:
            raise ValueError(f"Unknown coupling_mode {self.coupling_mode}")
