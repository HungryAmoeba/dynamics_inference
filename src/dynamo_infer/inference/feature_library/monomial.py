"""Monomial polynomial feature library."""

import itertools
import jax.numpy as jnp
import sympy as sp
import numpy as np
from typing import List, Optional, Tuple
from sklearn.utils.validation import check_is_fitted
from .base import BaseFeatureLibrary


class MonomialPolynomialLibrary(BaseFeatureLibrary):
    """
    Creates a library of polynomial features using the monomial basis.
    
    This library generates features of the form x_i^k for each input feature x_i
    and polynomial degrees k from 0 to degree.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum polynomial degree
    include_cross_terms : bool, default=False
        Whether to include cross terms (e.g., x_i * x_j for i != j)
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

    def fit(self, X, y=None) -> "MonomialPolynomialLibrary":
        """
        Fit the library to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or int
            Training data or number of features
        y : ignored
            Not used, present for sklearn compatibility
            
        Returns
        -------
        self : fitted library
        """
        if isinstance(X, int):
            # Called with just the number of features
            n_features = X
            self.n_features_in_ = n_features
            self.non_zero_idxs = tuple(range(n_features))
            self.constant_term_idxs = ()
            self.n_features_in_non_trivial = n_features
        else:
            X = self.validate_input(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_samples, n_features = X.shape

            # Identify which columns are nonzero
            non_zero_columns = jnp.any(X != 0, axis=0)
            mask_np = np.asarray(non_zero_columns)

            self.non_zero_idxs = tuple(int(i) for i in np.where(mask_np)[0])
            self.constant_term_idxs = tuple(int(i) for i in np.where(~mask_np)[0])
            self.n_features_in_ = n_features
            self.n_features_in_non_trivial = len(self.non_zero_idxs)

        # Calculate number of output features
        if self.include_cross_terms:
            if self.output_with_zeros:
                self.n_output_features_ = (self.degree + 1) ** self.n_features_in_
            else:
                self.n_output_features_ = (
                    self.degree + 1
                ) ** self.n_features_in_non_trivial
        else:
            if self.output_with_zeros:
                self.n_output_features_ = self.n_features_in_ * (self.degree + 1)
            else:
                self.n_output_features_ = self.n_features_in_non_trivial * (
                    self.degree + 1
                )

        return self

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform input data to monomial polynomial features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_output_features)
            Transformed features
        """
        check_is_fitted(self, ["non_zero_idxs", "constant_term_idxs"])
        X = self.validate_input(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples, _ = X.shape
        N = self.degree

        # Extract non-zero columns
        X_nz = X[:, self.non_zero_idxs]

        def eval_monomials(x_feat):
            """Evaluate monomials x^0, x^1, ..., x^N for a single feature."""
            return jnp.vander(x_feat, N + 1, increasing=True)  # (n_samples, N+1)

        # Evaluate monomials for each non-trivial feature
        univ = jnp.stack(
            [eval_monomials(X_nz[:, i]) for i in range(X_nz.shape[1])], axis=1
        )
        # shape: (n_samples, n_nontriv, N+1)

        # If we want to include originally zero columns
        if self.output_with_zeros:
            full = jnp.zeros((n_samples, self.n_features_in_, N + 1))
            full = full.at[:, self.non_zero_idxs, :].set(univ)
            # Set constant term = 1 for originally zero columns
            if self.constant_term_idxs:
                full = full.at[:, self.constant_term_idxs, 0].set(1.0)
            univ = full

        if self.include_cross_terms:
            # Generate all cross terms
            n_feats = univ.shape[1]
            powers = itertools.product(range(N + 1), repeat=n_feats)
            outs = []
            for p in powers:
                # For each multi-index p = (p0, p1, ...)
                terms = [univ[:, i, deg] for i, deg in enumerate(p)]
                outs.append(jnp.prod(jnp.stack(terms, axis=-1), axis=-1))
            return jnp.stack(outs, axis=-1)
        else:
            # Just flatten the last two dimensions (feature, degree)
            return univ.reshape(n_samples, -1)

    def get_feature_names_symbolic(self, feature_names=None) -> List[sp.Expr]:
        """
        Get symbolic expressions for the features.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names for input features. If None, uses d_0, d_1, ...
            
        Returns
        -------
        expressions : list of sympy expressions
            Symbolic expressions for each feature
        """
        check_is_fitted(self, ["non_zero_idxs", "constant_term_idxs"])
        N = self.degree
        n_feats = self.n_features_in_

        if feature_names is None:
            feature_names = [f"d_{i}" for i in range(n_feats)]

        poly_exprs = []
        for i in range(n_feats):
            x = sp.Symbol(feature_names[i])
            exprs = [x**j for j in range(N + 1)]
            # For originally zero columns, set all but constant term to 0
            if i in self.constant_term_idxs:
                exprs = [sp.S(1)] + [sp.S(0)] * N
            poly_exprs.append(exprs)

        if self.include_cross_terms:
            # Generate all combinations
            combos = itertools.product(*poly_exprs)
            return [sp.simplify(sp.Mul(*combo)) for combo in combos]
        else:
            # Flatten feature-major order
            return [expr for feat_exprs in poly_exprs for expr in feat_exprs]

    def get_feature_names(
        self, input_features: Optional[List[str]] = None, rounding: Optional[int] = 2
    ) -> List[str]:
        """
        Get string names for the features.
        
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


def create_monomial_library(degree: int = 2, **kwargs) -> MonomialPolynomialLibrary:
    """
    Factory function to create a monomial polynomial library.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum polynomial degree
    **kwargs : additional arguments
        Passed to MonomialPolynomialLibrary constructor
        
    Returns
    -------
    library : MonomialPolynomialLibrary
        Configured monomial library
    """
    return MonomialPolynomialLibrary(degree=degree, **kwargs)