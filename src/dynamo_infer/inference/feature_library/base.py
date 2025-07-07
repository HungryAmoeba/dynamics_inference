"""
Base class for feature library classes.
"""

import abc
import warnings
from functools import wraps
from itertools import repeat
from typing import Optional, Sequence, List, Dict, Any

import jax
import numpy as np
import jax.numpy as jnp
from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseFeatureLibrary(TransformerMixin):
    """
    Base class for feature libraries.

    Forces subclasses to implement ``fit``, ``transform``,
    and ``get_feature_names`` methods.
    """

    def __init__(self):
        self.n_features_in_: int = 0
        self.n_output_features_: int = 0

    def validate_input(self, x, *args, **kwargs):
        """Validate input data."""
        if isinstance(x, (list, tuple)):
            x = jnp.array(x)
        return jnp.asarray(x)

    # Force subclasses to implement this
    @abc.abstractmethod
    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def transform(self, x):
        """
        Transform data.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, [n_samples, n_output_features]
            The matrix of features, where n_output_features is the number
            of features generated from the combination of inputs.
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        raise NotImplementedError

    def get_feature_names_symbolic(self, input_features=None):
        """Return symbolic feature expressions (default implementation)."""
        return self.get_feature_names(input_features)

    def __add__(self, other):
        """Concatenate two libraries."""
        return ConcatLibrary([self, other])

    def __mul__(self, other):
        """Tensor product of two libraries."""
        return TensoredLibrary([self, other])

    def __rmul__(self, other):
        return TensoredLibrary([self, other])

    @property
    def size(self):
        """Get the number of output features."""
        check_is_fitted(self)
        return self.n_output_features_


class ConcatLibrary(BaseFeatureLibrary):
    """Concatenate multiple libraries into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    Attributes
    ----------
    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.
    """

    def __init__(self, libraries: list):
        super().__init__()
        self.libraries = libraries

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        x = self.validate_input(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        n_features = x.shape[-1]
        self.n_features_in_ = n_features

        # First fit all libs provided below
        fitted_libs = [lib.fit(x, y) for lib in self.libraries]

        # Calculate the sum of output features
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])

        # Save fitted libs
        self.libraries = fitted_libs

        return self

    def transform(self, x):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        x = self.validate_input(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        for lib in self.libraries:
            check_is_fitted(lib)

        feature_sets = [lib.transform(x) for lib in self.libraries]
        xp = jnp.concatenate(feature_sets, axis=-1)

        return xp

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        feature_names = []
        for lib in self.libraries:
            lib_feat_names = lib.get_feature_names(input_features)
            feature_names += lib_feat_names
        return feature_names


class TensoredLibrary(BaseFeatureLibrary):
    """Tensor multiple libraries together into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    inputs_per_library_ : Sequence of Sequences of ints (default None)
        list that specifies which input indexes should be passed as
        inputs for each of the individual feature libraries.
        length must equal the number of feature libraries.  Default is
        that all inputs are used for every library.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the numbers of output features for each of the
        libraries that were tensored together.
    """

    def __init__(
        self,
        libraries: list,
        inputs_per_library: Optional[Sequence[Sequence[int]]] = None,
    ):
        super().__init__()
        self.libraries = libraries
        self.inputs_per_library = inputs_per_library

    def _combinations(self, lib_i: jnp.ndarray, lib_j: jnp.ndarray) -> jnp.ndarray:
        """
        Compute combinations of the numerical libraries.

        Returns
        -------
        lib_full : All combinations of the numerical library terms.
        """
        # Outer product along feature dimension
        return jnp.einsum('...i,...j->...ij', lib_i, lib_j).reshape(lib_i.shape[:-1] + (-1,))

    def _name_combinations(self, lib_i, lib_j):
        """
        Compute combinations of the library feature names.

        Returns
        -------
        lib_full : All combinations of the library feature names.
        """
        lib_full = []
        for i in range(len(lib_i)):
            for j in range(len(lib_j)):
                lib_full.append(lib_i[i] + " " + lib_j[j])
        return lib_full

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        x = self.validate_input(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        n_features = x.shape[-1]
        self.n_features_in_ = n_features

        # If parameter is not set, use all the inputs
        if self.inputs_per_library is None:
            self.inputs_per_library = [list(range(n_features))] * len(self.libraries)

        # First fit all libs provided below
        fitted_libs = []
        for i, lib in enumerate(self.libraries):
            idxs = self.inputs_per_library[i]
            fitted_libs.append(lib.fit(x[..., idxs], y))

        # Calculate the product of output features
        output_sizes = [lib.n_output_features_ for lib in fitted_libs]
        self.n_output_features_ = 1
        for osize in output_sizes:
            self.n_output_features_ *= osize

        # Save fitted libs
        self.libraries = fitted_libs

        return self

    def transform(self, x):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        x = self.validate_input(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        check_is_fitted(self)

        # Get transformed features from each library
        transformed_features = []
        for i, lib in enumerate(self.libraries):
            idxs = self.inputs_per_library[i]
            xp_i = lib.transform(x[..., idxs])
            transformed_features.append(xp_i)

        # Compute tensor products
        if len(transformed_features) == 1:
            return transformed_features[0]
        
        result = transformed_features[0]
        for i in range(1, len(transformed_features)):
            result = self._combinations(result, transformed_features[i])

        return result

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        # Get feature names from each library
        lib_names = []
        for i, lib in enumerate(self.libraries):
            idxs = self.inputs_per_library[i]
            input_features_i = [input_features[j] for j in idxs]
            lib_names.append(lib.get_feature_names(input_features_i))

        # Compute name combinations
        if len(lib_names) == 1:
            return lib_names[0]
            
        result_names = lib_names[0]
        for i in range(1, len(lib_names)):
            result_names = self._name_combinations(result_names, lib_names[i])

        return result_names