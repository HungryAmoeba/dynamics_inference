import abc
import numpy as np
import matplotlib.pyplot as plt

class BaseInference(abc.ABC):
    """
    A base class for inference tasks, providing shared utilities and requiring specific methods 
    to be implemented by subclasses.
    """

    def __init__(self, *args, **kwargs):
        self.models = []
        self.data_names = kwargs.get('data_names', ['X', 'Y', 'Z'])

    @abc.abstractmethod
    def preprocess_data(self, data):
        """
        Preprocess the input data. This should be implemented by subclasses.

        Parameters
        ----------
        data : np.ndarray
            The input data to be preprocessed.

        Returns
        -------
        processed_data : np.ndarray
            The preprocessed data.
        targets : np.ndarray
            The target values for the data.
        """
        pass

    @abc.abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data. This should be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """
        Predict outcomes using the trained model. This should be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def print_equation(self):
        """
        Print the derived equation from the model. This should be implemented by subclasses.
        """
        pass
