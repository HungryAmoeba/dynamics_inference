from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import numpy as np
from src.models.base_inference import BaseInference

class LassoInference(BaseInference):
    """
    A concrete implementation of BaseInference for Lasso regression-based inference.
    """

    def __init__(self, alpha=0.1, max_degree=3, data_names=None, dt=1):
        super().__init__()
        self.alpha = alpha
        self.max_degree = max_degree
        self.data_names = data_names if data_names is not None else []
        self.feature_names = []
        self.dt = 1
        self.model = None

    def preprocess_data(self, X):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=self.max_degree, include_bias=True)
        X_flat = X.reshape(-1, X.shape[-1])
        X_poly = poly.fit_transform(X_flat)
        
        #self.feature_names = poly.get_feature_names_out(['X', 'Y', 'Z'])

        # add in sine and cosine features
        # Generate trigonometric features
        trig_features = np.hstack([np.sin(X_flat), np.cos(X_flat)])
        trig_feature_names = []
        for _, name in enumerate(self.data_names):
            trig_feature_names.append(f'sin({name})')
            trig_feature_names.append(f'cos({name})')
        # Combine polynomial and trigonometric features
        all_features = np.hstack([X_poly, trig_features])
        all_feature_names = np.concatenate([poly.get_feature_names_out(self.data_names), trig_feature_names])
        self.feature_names = all_feature_names

        # get the velocity targets
        velocities = np.diff(X, axis=0) / self.dt
        targets = velocities.reshape(-1, X.shape[-1])
        
        all_features = all_features[:-1]  # Exclude the last row to match the target shape

        return all_features, targets

    def fit(self, X, y):
        # fits a model for all dimensions?
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=self.alpha, max_iter=100000)
        lasso.fit(X, y)
        if self.model is not None: 
            print("Warning: This has already been fit, be careful.")
        self.model = lasso 


    def predict(self, X):
        # note that as this is currently written, it requires preprocessing to generate the polynomial and trigonometric features
        return self.model.predict(X)


    def print_equation(self):
        if len(self.models) == 0:
            print("No model has been trained yet.")
            return
        if len(self.models) != 1:
            for model in self.models:
                non_zero_indices = np.where(model.coef_ != 0)[0]
                equation = " + ".join(
                    f"{model.coef_[i]:.4f}*{self.feature_names[i]}"
                    for i in non_zero_indices
                )
                print(f"Derived equation: dX = {equation}")
        
        if len(self.models) == 1:
            model = self.models[0]
            for model_coeff, feature_name in zip(model.coef_, self.data_names):            
                non_zero_indices = np.where(model_coeff != 0)[0]
                equation = " + ".join(
                    f"{model_coeff[i]:.4f}*{self.feature_names[i]}"
                    for i in non_zero_indices
                )
                print(f"Derived equation: d{feature_name} = {equation}")
