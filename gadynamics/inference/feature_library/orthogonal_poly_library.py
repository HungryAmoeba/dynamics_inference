from itertools import chain
from math import comb
from typing import Iterator
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial
import itertools

import sympy as sp

from gadynamics.inference.utils.polynomial_fits import orth_poly_mc_fast

# from ..utils import AxesArray
# from ..utils import comprehend_axes
# from ..utils import wrap_axes
# from ..utils._axis_conventions import AX_COORD
from .base import BaseFeatureLibrary

# from .base import x_sequence_or_item

from typing import List, Optional


def get_symbolic_polynomials(coeffs: jnp.ndarray, varname: str = "x") -> List[sp.Expr]:
    """
    Convert monomial-basis coefficients to symbolic polynomials using SymPy.

    Parameters
    ----------
    coeffs : array-like of shape (N+1, N+1)
        coeffs[j, k] is the x^j coefficient of p_k(x).
    varname : str
        Symbolic name of the variable (e.g., 'x', 'x0', etc.)

    Returns
    -------
    poly_exprs : list of SymPy expressions
        poly_exprs[k] = p_k(x) as a SymPy expression
    """
    N = coeffs.shape[0] - 1
    x = sp.Symbol(varname)
    poly_exprs = []

    for k in range(N + 1):
        expr = sum(float(coeffs[j, k]) * x**j for j in range(N + 1))
        poly_exprs.append(sp.simplify(expr))

    return poly_exprs


class OrthogonalPolynomialLibrary(BaseFeatureLibrary):
    """
    Creates a library of orthogonal polynomial features, orthonormal
    w.r.t. the weighted inner product defined by the input data.
    """

    def __init__(
        self,
        degree: int = 2,
        include_cross_terms: bool = False,
        output_with_zeros: bool = True,
    ):
        super().__init__()
        self.degree = degree
        self.include_cross_terms = include_cross_terms
        self.output_with_zeros = output_with_zeros

        # will be set in fit():
        self.n_features_in_: int = 0
        self.n_output_features_: int = 0
        self.n_features_in_non_trivial: int = 0
        # these two will be tuples of ints:
        self.non_zero_idxs: tuple[int, ...] = ()
        self.constant_term_idxs: tuple[int, ...] = ()

    def fit(self, X: jnp.ndarray, y=None) -> "OrthogonalPolynomialLibrary":
        X = jnp.asarray(X)
        n_samples, n_features = X.shape
        N = self.degree

        # 1) find which input columns are nonzero anywhere
        non_zero_columns = jnp.any(X != 0, axis=0)  # shape (n_features,)
        mask_np = np.asarray(non_zero_columns)  # convert to NumPy bool array

        # 2) store the integer indices for static indexing
        self.non_zero_idxs = tuple(int(i) for i in np.where(mask_np)[0])
        self.constant_term_idxs = tuple(int(i) for i in np.where(~mask_np)[0])

        # 3) reduce X to only the non‑trivial columns
        X_nz = X[:, self.non_zero_idxs]  # shape (n_samples, n_nontriv)

        # 4) build orthonormal polynomials for each nonzero column
        #    `orth_poly_mc_fast(x, N)` returns (coeffs, evaluations)
        batched_orth_poly = vmap(
            partial(orth_poly_mc_fast, N=N), in_axes=1, out_axes=(0, 0)
        )
        coeffs, P_eval = batched_orth_poly(X_nz)
        # P_eval: (n_nontriv, n_samples, N+1)

        if jnp.isnan(P_eval).any() or jnp.isnan(coeffs).any():
            raise ValueError("NaNs in orthogonal polynomial evaluation!")

        # 5) stash results
        self.coeffs_ = coeffs  # (n_nontriv, N+1, N+1)
        P_eval = jnp.transpose(P_eval, (1, 0, 2))  # → (n_samples, n_nontriv, N+1)

        # 6) setup shape‐metadata
        self.n_features_in_ = n_features
        self.n_features_in_non_trivial = len(self.non_zero_idxs)

        if self.include_cross_terms:
            if self.output_with_zeros:
                self.n_output_features_ = (N + 1) ** n_features
            else:
                self.n_output_features_ = (N + 1) ** self.n_features_in_non_trivial
        else:
            if self.output_with_zeros:
                self.n_output_features_ = n_features * (N + 1)
            else:
                self.n_output_features_ = self.n_features_in_non_trivial * (N + 1)

        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        check_is_fitted(self, ["coeffs_", "non_zero_idxs", "constant_term_idxs"])
        X = jnp.asarray(X)
        n_samples, _ = X.shape
        N = self.degree

        # 1) pick out the non‑trivial columns
        X_nz = X[:, self.non_zero_idxs]  # shape (n_samples, n_nontriv)

        # 2) evaluate univariate polys
        def apply_feature(x_feat, coeff):
            # x_feat: (n_samples,), coeff: (N+1, N+1)
            V = jnp.vander(x_feat, N + 1, increasing=True)  # (n_samples, N+1)
            return V @ coeff  # (n_samples, N+1)

        univ = vmap(apply_feature, in_axes=(1, 0))(X_nz, self.coeffs_)
        # univ: (n_nontriv, n_samples, N+1) → transpose → (n_samples, n_nontriv, N+1)
        univ = jnp.transpose(univ, (1, 0, 2))

        # 3) if we want zero‐columns back
        if self.output_with_zeros:
            full = jnp.zeros((n_samples, self.n_features_in_, N + 1))
            # set the evaluated block
            full = full.at[:, self.non_zero_idxs, :].set(univ)
            # set constant term=1 in all originally‐zero dims
            if self.constant_term_idxs:
                full = full.at[:, self.constant_term_idxs, 0].set(1.0)
            univ = full  # now shape (n_samples, n_features_in_, N+1)

        # 4) assemble output
        if self.include_cross_terms:
            n_feats = univ.shape[1]
            powers = itertools.product(range(N + 1), repeat=n_feats)
            outs = []
            for p in powers:
                # for each multi‐index p = (p0,p1,...)
                terms = [univ[:, i, deg] for i, deg in enumerate(p)]
                outs.append(jnp.prod(jnp.stack(terms, axis=-1), axis=-1))
            return jnp.stack(outs, axis=-1)

        else:
            # just flatten last two dims
            return univ.reshape(n_samples, -1)

    def get_feature_names_symbolic(self, feature_names=None) -> List[sp.Expr]:
        check_is_fitted(self, ["coeffs_", "non_zero_idxs", "constant_term_idxs"])
        N = self.degree
        n_feats = self.n_features_in_

        # default input names d_0, d_1, …
        if feature_names is None:
            feature_names = [f"d_{i}" for i in range(n_feats)]

        # pad coeffs_ back into full shape (n_feats, N+1, N+1)
        if self.coeffs_.shape[0] != n_feats:
            padded = jnp.zeros((n_feats, self.coeffs_.shape[1], self.coeffs_.shape[2]))
            # set non‑trivial rows
            padded = padded.at[self.non_zero_idxs, :, :].set(self.coeffs_)
            # for originally‑zero dims, force constant term = 1
            if self.constant_term_idxs:
                padded = padded.at[self.constant_term_idxs, 0, 0].set(1.0)
        else:
            padded = self.coeffs_

        # build list of lists of sympy exprs, one list per feature
        poly_exprs: List[List[sp.Expr]] = []
        for i, var in enumerate(feature_names):
            # padded[i] has shape (N+1, N+1), rows = monomial degrees
            exprs = []
            for k in range(N + 1):
                # p_k(x) = sum_j coeffs[j,k] * x**j
                coeff_col = padded[i, :, k]
                x = sp.Symbol(var)
                poly = sum(float(c) * x**j for j, c in enumerate(coeff_col))
                exprs.append(sp.simplify(poly))
            poly_exprs.append(exprs)

        # assemble cross‐terms or flat
        if getattr(self, "include_cross_terms", False):
            combos = itertools.product(*poly_exprs)
            return [sp.simplify(sp.Mul(*combo)) for combo in combos]
        else:
            # flatten feature‑major
            return [expr for feats in poly_exprs for expr in feats]

    def get_feature_names(
        self, input_features: List[str] = None, rounding: Optional[int] = 2
    ) -> List[str]:
        """
        Returns string names for each generated feature.
        """
        names = self.get_feature_names_symbolic(feature_names=input_features)
        if rounding is not None:
            names = [str(sp.N(expr, rounding)) for expr in names]
        else:
            names = [str(expr) for expr in names]
        return names


if __name__ == "__main__":
    # Example usage
    X = np.array([[1], [2], [3]])
    poly_lib = UnivariateOrthogonalPolynomialLibrary(degree=2)
    poly_lib.fit(X)
    transformed_X = poly_lib.transform(X)
    print(transformed_X)
