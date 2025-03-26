import pytest 
from src.models.bayesian_fitting import SparseBayesianMC
import numpy as np
import torch 
from src.data.toy_examples import simple_2d_sde_dataset, simple_3d_sde_dataset


def test_sparse_bayesian_2d():
    # get the 2d dataset 
    X, t = simple_2d_sde_dataset()
    reg = SparseBayesianMC(alpha=0.1, max_degree=5, data_names=['X', 'Y'], dt = 1)
    feats, targets = reg.preprocess_data(X)
    reg.fit(feats, targets)
    #reg.print_equation()
    print('Done fitting 2D model')


if __name__ == "__main__":
    test_sparse_bayesian_2d()