import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import gaussian_kde

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import sympy as sp


def print_polynomial_equations(polynomials):
    x = sp.symbols("x")
    for i, poly in enumerate(polynomials):
        poly_expr = sp.simplify(sp.lambdify(x, poly(x))(x))
        print(f"P_{i}(x) = {poly_expr}")


def orth_poly_mc(data_locations, N):
    """
    Generate a list of orthogonal polynomials using Monte Carlo integration.
    Parameters:
    data_locations (array-like): Locations where the polynomials will be evaluated.
    N (int): The highest degree of the orthogonal polynomials to generate.
    Returns:
    list: A list of functions representing the orthogonal polynomials.
    Example:
    >>> data_locations = np.random.uniform(-1, 1, 1000)
    >>> polys = orth_poly_mc(data_locations, 3)
    >>> for p in polys:
    >>>     print(p(0.5))
    """

    def inner_product(f, g, data_locations):
        return np.mean(f(data_locations) * g(data_locations))

    def normalize(f, data_locations):
        norm = np.sqrt(inner_product(f, f, data_locations))
        return lambda x: f(x) / norm

    def constant_poly(x):
        return x * 0 + 1

    polynomials = [normalize(constant_poly, data_locations)]

    for k in range(1, N + 1):

        def x_poly(x, k=k):
            return x * polynomials[k - 1](x)

        for j in range(k):
            proj_coeff = inner_product(
                x_poly, polynomials[j], data_locations
            ) / inner_product(polynomials[j], polynomials[j], data_locations)

            def new_x_poly(x, poly=x_poly, pj=polynomials[j], c=proj_coeff):
                return poly(x) - c * pj(x)

            x_poly = new_x_poly

        polynomials.append(normalize(x_poly, data_locations))
    return polynomials


def fit_polynomial_mc(X, Y, poly_list):
    def inner_product(poly, Y, data_locations):
        return np.mean(poly(data_locations) * Y)

    def least_squares_fit(X, Y, poly_list):
        alpha = np.zeros(len(poly_list))
        for k, poly in enumerate(poly_list):
            alpha[k] = inner_product(poly, Y, X)

        def P_star(x):
            return sum(alpha[k] * poly(x) for k, poly in enumerate(poly_list))

        return P_star

    return least_squares_fit(X, Y, poly_list)


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


def orth_poly_mc_fast_2(data_locations, N):
    x = jnp.asarray(data_locations)
    K = x.size

    # 1) Vandermonde
    V = jnp.vander(x, N + 1, increasing=True)
    Vw = V / jnp.sqrt(K)

    # 2) economy QR
    _, R = jnp.linalg.qr(Vw, mode="reduced")

    # 3) solve R @ A = I  (avoids explicit inversion)
    I = jnp.eye(N + 1)
    A = solve_triangular(R, I, lower=False)

    return A  # same shape (N+1,N+1); A[:,k] are the monomial coefs of p_k


def orth_poly_mc_fast_3(data_locations, N):
    x = jnp.asarray(data_locations)
    K = x.size

    # 1) Build weighted Vandermonde
    V = jnp.vander(x, N + 1, increasing=True)  # (K, N+1)
    Vw = V / jnp.sqrt(K)

    # 2) Thin QR
    Q, R = jnp.linalg.qr(Vw, mode="reduced")

    # 3) Flip any diagonal of R that is negative:
    diag_signs = jnp.sign(jnp.diag(R))
    diag_signs = jnp.where(diag_signs == 0, 1.0, diag_signs)
    # Multiply column‑n of Q and row‑n of R by diag_signs[n]:
    Q = Q * diag_signs[None, :]
    R = diag_signs[:, None] * R

    # 4a) evaluation at the data points
    P_eval = jnp.sqrt(K) * Q  # (K, N+1)

    # 4b) monomial coefficients
    coeffs = jnp.linalg.inv(R)  # (N+1, N+1)

    return coeffs, P_eval


def orth_poly(w_func, N, domain=[-1, 1]):
    """
    Generate a list of orthogonal polynomials with respect to a given weight function.
    Parameters:
    w_func (function): Weight function w(x) used in the inner product.
    N (int): The highest degree of the orthogonal polynomials to generate.
    domain (list, optional): The integration domain for the inner product, default is [-1, 1].
    Returns:
    list: A list of functions representing the orthogonal polynomials.
    Example:
    >>> w_func = lambda x: 1
    >>> polys = orth_poly(w_func, 3)
    >>> for p in polys:
    >>>     print(p(0.5))
    """

    def inner_product(f, g):
        integrand = lambda x: w_func(x) * f(x) * g(x)
        result, _ = quad(integrand, domain[0], domain[1])
        return result

    def normalize(f):
        norm = np.sqrt(inner_product(f, f))
        return lambda x: f(x) / norm

    # Define the orthogonal polynomials
    def constant_poly(x):
        return sp.poly(x)

    polynomials = [normalize(constant_poly)]

    for k in range(1, N + 1):

        def x_poly(x, k=k):
            return x * polynomials[k - 1](x)

        for j in range(k):
            proj_coeff = inner_product(x_poly, polynomials[j]) / inner_product(
                polynomials[j], polynomials[j]
            )

            def new_x_poly(x, poly=x_poly, pj=polynomials[j], c=proj_coeff):
                return poly(x) - c * pj(x)

            x_poly = new_x_poly

        polynomials.append(normalize(x_poly))

    return polynomials


def plot_polynomials(polynomials, w_func, N, domain=[-1, 1]):
    x = np.linspace(domain[0], domain[1], 1000)
    w = w_func(x)

    for i in range(N + 1):
        plt.plot(x, polynomials[i](x), label=f"P_{i}")

    plt.title("Orthogonal polynomials on [-1,1] wrt w = exp(pi*x)")
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.legend()
    plt.show()


def test_orthogonality(polynomials, w_func, N, domain=[-1, 1]):
    def inner_product(f, g):
        integrand = lambda x: w_func(x) * f(x) * g(x)
        result, _ = quad(integrand, domain[0], domain[1])
        return result

    I = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            I[i, j] = inner_product(polynomials[i], polynomials[j])

    err = np.linalg.norm(I - np.eye(N + 1))
    print(f"Orthogonality error: {err}")


def least_squares_approximation(polynomials, w_func, f, N, domain=[-1, 1]):
    def inner_product(f, g):
        integrand = lambda x: w_func(x) * f(x) * g(x)
        result, _ = quad(integrand, domain[0], domain[1])
        return result

    alpha = np.zeros(N + 1)
    for k in range(N + 1):
        alpha[k] = inner_product(f, polynomials[k])

    def P_star(x):
        return sum(alpha[k] * polynomials[k](x) for k in range(N + 1))

    x = np.linspace(domain[0], domain[1], 1000)
    plt.plot(x, f(x), "b", label="f(x) = |x|")
    plt.plot(x, P_star(x), "--r", label="P* approximation")
    plt.title("Least-squares approximation to |x| wrt w = exp(pi*x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Function to create a KDE-based weight function compatible with orth_poly
def kde_weight_function(data, domain, bw_method="scott"):
    # Create the KDE
    kde = gaussian_kde(data, bw_method=bw_method)

    # Normalize the KDE to ensure it's a valid weight function over the domain
    normalization_factor = np.trapz(kde(domain), domain)

    def w_func(x):
        return kde(x) / normalization_factor

    return w_func
