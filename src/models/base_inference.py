import abc
import matplotlib.pyplot as plt


class BaseInference(abc.ABC):
    """
    A base class for inference tasks, providing shared utilities and requiring specific methods
    to be implemented by subclasses.
    """

    def __init__(self, *args, **kwargs):
        self.models = []
        self.data_names = kwargs.get("data_names", ["X", "Y", "Z"])

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

    def plot_trajectory(self, actual, predicted, title="Trajectory Comparison"):
        """
        Utility function to plot actual vs predicted trajectories.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="Actual", color="blue")
        ax.plot(
            predicted[:, 0],
            predicted[:, 1],
            predicted[:, 2],
            label="Predicted",
            color="red",
            linestyle="dashed",
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()
