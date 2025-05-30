import itertools
import jax.numpy as jnp
import sympy as sp
import numpy as np
from typing import List, Optional
from sklearn.utils.validation import check_is_fitted
from gadynamics.inference.feature_library.base import BaseFeatureLibrary


class MonomialPolynomialLibrary(BaseFeatureLibrary):
    """
    Creates a library of polynomial features using the monomial basis.
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
        if isinstance(X, int):
            n_features = X
            self.n_features_in_ = n_features
            self.non_zero_idxs = tuple(range(n_features))
            self.constant_term_idxs = ()
            self.n_features_in_non_trivial = n_features
        else:
            X = jnp.asarray(X)
            n_samples, n_features = X.shape

            # Identify which columns are nonzero
            non_zero_columns = jnp.any(X != 0, axis=0)
            mask_np = np.asarray(non_zero_columns)

            self.non_zero_idxs = tuple(int(i) for i in np.where(mask_np)[0])
            self.constant_term_idxs = tuple(int(i) for i in np.where(~mask_np)[0])
            self.n_features_in_ = n_features
            self.n_features_in_non_trivial = len(self.non_zero_idxs)

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
        check_is_fitted(self, ["non_zero_idxs", "constant_term_idxs"])
        X = jnp.asarray(X)
        n_samples, _ = X.shape
        N = self.degree

        X_nz = X[:, self.non_zero_idxs]

        def eval_monomials(x_feat):
            return jnp.vander(x_feat, N + 1, increasing=True)  # (n_samples, N+1)

        univ = jnp.stack(
            [eval_monomials(X_nz[:, i]) for i in range(X_nz.shape[1])], axis=1
        )
        # shape: (n_samples, n_nontriv, N+1)

        if self.output_with_zeros:
            full = jnp.zeros((n_samples, self.n_features_in_, N + 1))
            full = full.at[:, self.non_zero_idxs, :].set(univ)
            if self.constant_term_idxs:
                full = full.at[:, self.constant_term_idxs, 0].set(1.0)
            univ = full

        if self.include_cross_terms:
            n_feats = univ.shape[1]
            powers = itertools.product(range(N + 1), repeat=n_feats)
            outs = []
            for p in powers:
                terms = [univ[:, i, deg] for i, deg in enumerate(p)]
                outs.append(jnp.prod(jnp.stack(terms, axis=-1), axis=-1))
            return jnp.stack(outs, axis=-1)
        else:
            return univ.reshape(n_samples, -1)

    def get_feature_names_symbolic(self, feature_names=None) -> List[sp.Expr]:
        check_is_fitted(self, ["non_zero_idxs", "constant_term_idxs"])
        N = self.degree
        n_feats = self.n_features_in_

        if feature_names is None:
            feature_names = [f"d_{i}" for i in range(n_feats)]

        poly_exprs = []
        for i in range(n_feats):
            x = sp.Symbol(feature_names[i])
            exprs = [x**j for j in range(N + 1)]
            if i in self.constant_term_idxs:
                exprs = [sp.S(1)] + [sp.S(0)] * N
            poly_exprs.append(exprs)

        if self.include_cross_terms:
            combos = itertools.product(*poly_exprs)
            return [sp.simplify(sp.Mul(*combo)) for combo in combos]
        else:
            return [expr for feat_exprs in poly_exprs for expr in feat_exprs]

    def get_feature_names(
        self, input_features: List[str] = None, rounding: Optional[int] = 2
    ) -> List[str]:
        names = self.get_feature_names_symbolic(input_features)
        if rounding is not None:
            names = [str(sp.N(expr, rounding)) for expr in names]
        else:
            names = [str(expr) for expr in names]
        return names


if __name__ == "__main__":
    poly_lib = MonomialPolynomialLibrary(degree=2, include_cross_terms=False)
    X = jnp.array([[1, 2], [3, 4], [5, 6]])
    poly_lib.fit(X)
    transformed = poly_lib.transform(X)
    feature_names = poly_lib.get_feature_names()
    print("Transformed Features:\n", transformed)
