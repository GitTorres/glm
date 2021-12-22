import pandas as pd
import sklearn.datasets as datasets
import numpy as np
from simulate import Simulate
from numpy import ndarray
from pandas import DataFrame
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm
import math
from typing import Tuple

# def compare_poisson() -> None:
#     poisson_model: Simulate = Simulate(dist='poisson')
#     poisson_model.generate_model_data(n=1000)

#     glm: GLE = GLE(solver='IRLSM',
#                     family= "poisson",
#                     lambda_ = 0,
#                     compute_p_values = True,
#                     standardize = True,
#                     intercept = True,
#                     link = 'Log')

#     glm.train(x = poisson_model.predictors,
#             y = poisson_model.response,
#             training_frame = H2OFrame(poisson_model.data))

#     coefficients = pd.DataFrame({'true': poisson_model.coeffs,
#                                  'estimated': glm._model_json['output']['coefficients_table']['coefficients']})

#     print(coefficients)

#     return None


def compare_sklearn(features: int) -> None:
    X: ndarray
    y: ndarray
    coef_sklearn: ndarray

    X, y, coef_sklearn = datasets.make_regression(
        n_samples=3000000, n_features=features, n_informative=5, noise=1, coef=True
    )
    data: DataFrame = pd.DataFrame(
        data=np.concatenate((X, y.reshape(len(y), 1)), axis=1),
        columns=[*[f"feat_{i+1}" for i in range(features)], "y"],
    )
    model = sm.OLS(y, X).fit()

    results = {
        "model": model,
        "dataframe": data,
        "input_matrix": X,
        "computed_y": np.dot(X, coef_sklearn),
        "true_y": y,
        "fitted_y": model.predict(),
        "true_coef": coef_sklearn,
    }

    return results


def corr(x, y):
    """
    Return the sample correlation of x and y.
    """
    z_x = (x - x.mean()) / x.std()
    z_y = (y - y.mean()) / y.std()

    # return corr(x,y) = covariance(z_x, z_y)
    return (z_x * z_y).sum() / (len(x) - 1)


def correlated_norm_rvs(corr):
    N = 300000
    x1: ndarray = norm.rvs(loc=0, scale=1, size=N)
    x2: ndarray = norm.rvs(loc=0, scale=1, size=N)
    x3: ndarray = corr * x1 + math.sqrt(1 - corr ** 2) * x2

    return x1, x3


