import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import gaussian_kde

import sympy as sp

def print_polynomial_equations(polynomials):
    x = sp.symbols('x')
    for i, poly in enumerate(polynomials):
        poly_expr = sp.simplify(sp.lambdify(x, poly(x))(x))
        print(f"P_{i}(x) = {poly_expr}")

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
        return np.ones_like(x)
    
    polynomials = [normalize(constant_poly)]
    
    for k in range(1, N + 1):
        def x_poly(x, k=k):
            return x * polynomials[k-1](x)
        
        for j in range(k):
            proj_coeff = inner_product(x_poly, polynomials[j]) / inner_product(polynomials[j], polynomials[j])
            def new_x_poly(x, poly=x_poly, pj=polynomials[j], c=proj_coeff):
                return poly(x) - c * pj(x)
            x_poly = new_x_poly
        
        polynomials.append(normalize(x_poly))

    return polynomials

def plot_polynomials(polynomials, w_func, N, domain=[-1, 1]):
    x = np.linspace(domain[0], domain[1], 1000)
    w = w_func(x)
    
    for i in range(N + 1):
        plt.plot(x, polynomials[i](x), label=f'P_{i}')
    
    plt.title('Orthogonal polynomials on [-1,1] wrt w = exp(pi*x)')
    plt.xlabel('x')
    plt.ylabel('P(x)')
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
    plt.plot(x, f(x), 'b', label='f(x) = |x|')
    plt.plot(x, P_star(x), '--r', label='P* approximation')
    plt.title('Least-squares approximation to |x| wrt w = exp(pi*x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Function to create a KDE-based weight function compatible with orth_poly
def kde_weight_function(data, domain, bw_method='scott'):
    # Create the KDE
    kde = gaussian_kde(data, bw_method=bw_method)

    # Normalize the KDE to ensure it's a valid weight function over the domain
    normalization_factor = np.trapz(kde(domain), domain)
    
    def w_func(x):
        return kde(x) / normalization_factor
    
    return w_func