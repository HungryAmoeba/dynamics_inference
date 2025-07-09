"""Polynomial feature library (alias for monomial library)."""

from .monomial import MonomialPolynomialLibrary


class PolynomialLibrary(MonomialPolynomialLibrary):
    """
    Polynomial feature library.
    
    This is an alias for MonomialPolynomialLibrary for backward compatibility.
    Creates polynomial features using the monomial basis.
    """
    pass


def create_polynomial_library(degree: int = 2, **kwargs) -> PolynomialLibrary:
    """
    Factory function to create a polynomial library.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum polynomial degree
    **kwargs : additional arguments
        Passed to PolynomialLibrary constructor
        
    Returns
    -------
    library : PolynomialLibrary
        Configured polynomial library
    """
    return PolynomialLibrary(degree=degree, **kwargs)