from .feature_library.orthogonal_poly_library import (
    UnivariateOrthogonalPolynomialLibrary,
)
from dynamics_inference.inference.GA_inference_scikit import GA_Dynamics_Inference

import jax
import jax.numpy as jnp

import sys

# Add the parent directory to the path
sys.path.append("..")

num_features = 5
num_samples = 100
X = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_samples, num_features))


poly_lib = UnivariateOrthogonalPolynomialLibrary(degree=3, include_cross_terms=False)
poly_lib.fit(X)
transformed_X = poly_lib.transform(X)
assert jnp.allclose(
    transformed_X, poly_lib.P_eval_, atol=1e-4
), "Transformed data does not match the expected output."

print(poly_lib.get_feature_names())

del poly_lib
del transformed_X

# now test cross terms
# please don't break

poly_lib = UnivariateOrthogonalPolynomialLibrary(degree=2, include_cross_terms=True)
poly_lib.fit(X)
transformed_X = poly_lib.transform(X)
print(transformed_X.shape)
assert (
    transformed_X.shape[1] == (poly_lib.degree + 1) ** num_features
), "Transformed data does not match the expected output."
print(poly_lib.get_feature_names())

model = GA_Dynamics_Inference(
    feature_library=poly_lib,
    num_features=num_features,
    num_samples=num_samples,
    include_cross_terms=True,
)
