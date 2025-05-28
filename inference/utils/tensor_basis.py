import jax.numpy as jnp


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
