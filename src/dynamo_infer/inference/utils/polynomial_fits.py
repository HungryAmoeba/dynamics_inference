"""Polynomial fitting utilities for orthogonal polynomials."""

import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import sympy as sp
from jax import vmap
from functools import partial
from typing import List, Tuple


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


def orth_poly_mc_fast(data_locations, N):
    """
    Generate orthonormal polynomials p_0…p_N w.r.t. the empirical distribution
    on `data_locations`, returning both:

      • coeffs: monomial‐basis coefficients (shape (N+1,N+1))
                coeffs[j, k] is the x^j coefficient of p_k(x).
      • P_eval: evaluations p_k(x_i) (shape (K, N+1)) such that
                (1/K) * P_eval.T @ P_eval = I.

    Parameters
    ----------
    data_locations : array-like, shape (K,)
        Sample points x_i.
    N : int
        Highest polynomial degree.

    Returns
    -------
    coeffs : jnp.ndarray, shape (N+1, N+1)
        coeffs[:, k] are the monomial coefficients of p_k.
    P_eval : jnp.ndarray, shape (K, N+1)
        P_eval[i, k] = p_k(x_i), orthonormal under the discrete mean inner‐product.
    """
    x = jnp.asarray(data_locations)
    K = x.size

    # 1) Vandermonde in monomial basis: V[i,j] = x[i]**j
    V = jnp.vander(x, N + 1, increasing=True)  # shape (K, N+1)

    # 2) Incorporate weight w = 1/K by scaling rows by 1/sqrt K
    Vw = V / jnp.sqrt(K)

    # 3) Thin QR: Vw = Q @ R,    Q^T Q = I
    Q, R = jnp.linalg.qr(Vw, mode="reduced")  # Q: (K, N+1), R: (N+1, N+1)

    # 4a) To get evaluations p_k(x_i) satisfying (1/K) sum p_m p_n = \delta_mn:
    #    rescale Q by sqrt K:
    P_eval = jnp.sqrt(K) * Q  # shape (K, N+1)

    # 4b) To get monomial‐basis coeffs of each p_k, solve R @ a_k = e_k:
    invR = jnp.linalg.inv(R)  # (N+1, N+1)
    coeffs = invR  # coeffs[:, k] = a_k

    return coeffs, P_eval


def orth_poly_mc_fast_stable(data_locations, N):
    """
    Generate orthonormal polynomials with improved numerical stability.
    
    Uses solve_triangular instead of matrix inversion for better stability.
    """
    x = jnp.asarray(data_locations)
    K = x.size

    # 1) Build weighted Vandermonde
    V = jnp.vander(x, N + 1, increasing=True)  # (K, N+1)
    Vw = V / jnp.sqrt(K)

    # 2) Thin QR
    Q, R = jnp.linalg.qr(Vw, mode="reduced")

    # 3) Flip any diagonal of R that is negative for consistency:
    diag_signs = jnp.sign(jnp.diag(R))
    diag_signs = jnp.where(diag_signs == 0, 1.0, diag_signs)
    # Multiply column‑n of Q and row‑n of R by diag_signs[n]:
    Q = Q * diag_signs[None, :]
    R = diag_signs[:, None] * R

    # 4a) evaluation at the data points
    P_eval = jnp.sqrt(K) * Q  # (K, N+1)

    # 4b) monomial coefficients using stable triangular solve
    I = jnp.eye(N + 1)
    coeffs = solve_triangular(R, I, lower=False)  # (N+1, N+1)

    return coeffs, P_eval


# Vectorized version for multiple features
def orth_poly_mc_fast_batch(data_locations_batch, N):
    """
    Compute orthogonal polynomials for multiple features in parallel.
    
    Parameters
    ----------
    data_locations_batch : array-like, shape (n_features, n_samples)
        Batch of data locations for different features
    N : int
        Highest polynomial degree
        
    Returns
    -------
    coeffs_batch : array-like, shape (n_features, N+1, N+1)
        Coefficients for each feature
    P_eval_batch : array-like, shape (n_features, n_samples, N+1)
        Evaluations for each feature
    """
    batched_orth_poly = vmap(
        partial(orth_poly_mc_fast_stable, N=N), in_axes=0, out_axes=(0, 0)
    )
    return batched_orth_poly(data_locations_batch)


def evaluate_orthogonal_polynomials(x, coeffs):
    """
    Evaluate orthogonal polynomials at new points.
    
    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Points to evaluate at
    coeffs : array-like, shape (N+1, N+1)
        Polynomial coefficients from orth_poly_mc_fast
        
    Returns
    -------
    P_eval : array-like, shape (n_samples, N+1)
        Polynomial evaluations
    """
    x = jnp.asarray(x)
    N = coeffs.shape[0] - 1
    
    # Build Vandermonde matrix
    V = jnp.vander(x, N + 1, increasing=True)
    
    # Apply coefficients to get polynomial evaluations
    P_eval = V @ coeffs
    
    return P_eval