import pandas as pd
import sklearn.datasets as datasets
import numpy as np
from utils.simulate import Simulate
from numpy import ndarray
from pandas import DataFrame
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

    X, y, coef_sklearn = \
        datasets.make_regression(
            n_samples=3000000, 
            n_features=features, 
            n_informative=10, 
            noise=0, 
            coef=True)
    data: DataFrame = \
        pd.DataFrame(
            data=np.concatenate(
                    (X, y.reshape(len(y),1)), 
                    axis=1), 
            columns=[*[f'feat_{i+1}' for i in range(features)], 'y'])
    model = sm.OLS(y, X).fit()

    results = {
        'model': model,
        'dataframe': data,
        'input_matrix': X,
        'computed_y': np.dot(X, coef_sklearn),
        'true_y': y,
        'fitted_y': model.predict(),
        'true_coef': coef_sklearn
    }

    return results