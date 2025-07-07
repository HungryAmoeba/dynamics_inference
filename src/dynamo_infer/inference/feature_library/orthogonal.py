"""Orthogonal polynomial feature library."""

import itertools
from typing import List, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from jax import vmap
from functools import partial
import sympy as sp
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from ..utils.polynomial_fits import orth_poly_mc_fast, get_symbolic_polynomials


class OrthogonalPolynomialLibrary(BaseFeatureLibrary):
    """
    Creates a library of orthogonal polynomial features, orthonormal
    w.r.t. the weighted inner product defined by the input data.
    
    This library generates orthogonal polynomials fitted to the empirical
    distribution of the input data. This can provide better numerical
    stability and conditioning compared to monomial polynomials.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum polynomial degree
    include_cross_terms : bool, default=False
        Whether to include cross terms between different input features
    output_with_zeros : bool, default=True
        Whether to include zero columns in output for originally zero features
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

        # Will be set in fit():
        self.n_features_in_non_trivial: int = 0
        self.non_zero_idxs: Tuple[int, ...] = ()
        self.constant_term_idxs: Tuple[int, ...] = ()
        self.coeffs_: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray, y=None) -> "OrthogonalPolynomialLibrary":
        """
        Fit orthogonal polynomials to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : ignored
            Not used, present for sklearn compatibility
            
        Returns
        -------
        self : fitted library
        """
        X = self.validate_input(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        N = self.degree

        # 1) Find which input columns are nonzero anywhere
        non_zero_columns = jnp.any(X != 0, axis=0)  # shape (n_features,)
        mask_np = np.asarray(non_zero_columns)  # convert to NumPy bool array

        # 2) Store the integer indices for static indexing
        self.non_zero_idxs = tuple(int(i) for i in np.where(mask_np)[0])
        self.constant_term_idxs = tuple(int(i) for i in np.where(~mask_np)[0])

        # 3) Reduce X to only the non‑trivial columns
        X_nz = X[:, self.non_zero_idxs]  # shape (n_samples, n_nontriv)

        # 4) Build orthonormal polynomials for each nonzero column
        #    `orth_poly_mc_fast(x, N)` returns (coeffs, evaluations)
        batched_orth_poly = vmap(
            partial(orth_poly_mc_fast, N=N), in_axes=1, out_axes=(0, 0)
        )
        coeffs, P_eval = batched_orth_poly(X_nz)
        # P_eval: (n_nontriv, n_samples, N+1)

        if jnp.isnan(P_eval).any() or jnp.isnan(coeffs).any():
            raise ValueError("NaNs in orthogonal polynomial evaluation!")

        # 5) Stash results
        self.coeffs_ = coeffs  # (n_nontriv, N+1, N+1)
        P_eval = jnp.transpose(P_eval, (1, 0, 2))  # → (n_samples, n_nontriv, N+1)

        # 6) Setup shape‐metadata
        self.n_features_in_ = n_features
        self.n_features_in_non_trivial = len(self.non_zero_idxs)

        # Calculate number of output features
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
        """
        Transform input data using fitted orthogonal polynomials.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_output_features)
            Transformed features
        """
        check_is_fitted(self, ["coeffs_", "non_zero_idxs", "constant_term_idxs"])
        X = self.validate_input(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, _ = X.shape
        N = self.degree

        # 1) Pick out the non‑trivial columns
        X_nz = X[:, self.non_zero_idxs]  # shape (n_samples, n_nontriv)

        # 2) Evaluate univariate polynomials
        def apply_feature(x_feat, coeff):
            # x_feat: (n_samples,), coeff: (N+1, N+1)
            V = jnp.vander(x_feat, N + 1, increasing=True)  # (n_samples, N+1)
            return V @ coeff  # (n_samples, N+1)

        univ = vmap(apply_feature, in_axes=(1, 0))(X_nz, self.coeffs_)
        # univ: (n_nontriv, n_samples, N+1) → transpose → (n_samples, n_nontriv, N+1)
        univ = jnp.transpose(univ, (1, 0, 2))

        # 3) If we want zero‐columns back
        if self.output_with_zeros:
            full = jnp.zeros((n_samples, self.n_features_in_, N + 1))
            # Set the evaluated block
            full = full.at[:, self.non_zero_idxs, :].set(univ)
            # Set constant term=1 in all originally‐zero dims
            if self.constant_term_idxs:
                full = full.at[:, self.constant_term_idxs, 0].set(1.0)
            univ = full  # now shape (n_samples, n_features_in_, N+1)

        # 4) Assemble output
        if self.include_cross_terms:
            n_feats = univ.shape[1]
            powers = itertools.product(range(N + 1), repeat=n_feats)
            outs = []
            for p in powers:
                # For each multi‐index p = (p0,p1,...)
                terms = [univ[:, i, deg] for i, deg in enumerate(p)]
                outs.append(jnp.prod(jnp.stack(terms, axis=-1), axis=-1))
            return jnp.stack(outs, axis=-1)
        else:
            # Just flatten last two dims
            return univ.reshape(n_samples, -1)

    def get_feature_names_symbolic(self, feature_names=None) -> List[sp.Expr]:
        """
        Get symbolic expressions for the orthogonal polynomial features.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names for input features. If None, uses d_0, d_1, ...
            
        Returns
        -------
        expressions : list of sympy expressions
            Symbolic expressions for each feature
        """
        check_is_fitted(self, ["coeffs_", "non_zero_idxs", "constant_term_idxs"])
        N = self.degree
        n_feats = self.n_features_in_

        # Default input names d_0, d_1, …
        if feature_names is None:
            feature_names = [f"d_{i}" for i in range(n_feats)]

        # Pad coeffs_ back into full shape (n_feats, N+1, N+1)
        if self.coeffs_.shape[0] != n_feats:
            padded = jnp.zeros((n_feats, self.coeffs_.shape[1], self.coeffs_.shape[2]))
            # Set non‑trivial rows
            padded = padded.at[self.non_zero_idxs, :, :].set(self.coeffs_)
            # For originally‑zero dims, force constant term = 1
            if self.constant_term_idxs:
                padded = padded.at[self.constant_term_idxs, 0, 0].set(1.0)
        else:
            padded = self.coeffs_

        # Build list of lists of sympy exprs, one list per feature
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

        # Assemble cross‐terms or flat
        if self.include_cross_terms:
            combos = itertools.product(*poly_exprs)
            return [sp.simplify(sp.Mul(*combo)) for combo in combos]
        else:
            # Flatten feature‐major
            return [expr for feats in poly_exprs for expr in feats]

    def get_feature_names(
        self, input_features: Optional[List[str]] = None, rounding: Optional[int] = 2
    ) -> List[str]:
        """
        Get string names for the orthogonal polynomial features.
        
        Parameters
        ----------
        input_features : list of str, optional
            Names for input features
        rounding : int, optional
            Number of decimal places for rounding numeric coefficients
            
        Returns
        -------
        feature_names : list of str
            String names for each feature
        """
        symbolic_names = self.get_feature_names_symbolic(input_features)
        if rounding is not None:
            names = [str(sp.N(expr, rounding)) for expr in symbolic_names]
        else:
            names = [str(expr) for expr in symbolic_names]
        return names

    def get_polynomial_info(self) -> dict:
        """
        Get information about the fitted orthogonal polynomials.
        
        Returns
        -------
        info : dict
            Dictionary containing polynomial information
        """
        if not hasattr(self, 'coeffs_') or self.coeffs_ is None:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "degree": self.degree,
            "n_features": self.n_features_in_,
            "n_nonzero_features": self.n_features_in_non_trivial,
            "n_output_features": self.n_output_features_,
            "include_cross_terms": self.include_cross_terms,
            "coeffs_shape": self.coeffs_.shape,
            "non_zero_indices": self.non_zero_idxs,
        }


def create_orthogonal_library(degree: int = 2, **kwargs) -> OrthogonalPolynomialLibrary:
    """
    Factory function to create an orthogonal polynomial library.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum polynomial degree
    **kwargs : additional arguments
        Passed to OrthogonalPolynomialLibrary constructor
        
    Returns
    -------
    library : OrthogonalPolynomialLibrary
        Configured orthogonal polynomial library
    """
    return OrthogonalPolynomialLibrary(degree=degree, **kwargs)