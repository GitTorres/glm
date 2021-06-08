import h2o
import pandas as pd
from h2o.frame import H2OFrame
from h2o.backend.cluster import H2OCluster
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as GLE
import sklearn.datasets as datasets
import numpy as np
from utils.simulate import Simulate

def compare_poisson() -> None:
    poisson_model: Simulate = Simulate(dist='poisson')
    poisson_model.generate_model_data(n=1000)

    glm: GLE = GLE(solver='IRLSM',
                    family= "poisson",
                    lambda_ = 0,
                    compute_p_values = True,
                    standardize = True,
                    intercept = True,
                    link = 'Log')

    glm.train(x = poisson_model.predictors, 
            y = poisson_model.response, 
            training_frame = H2OFrame(poisson_model.data))

    coefficients = pd.DataFrame({'true': poisson_model.coeffs,
                                 'estimated': glm._model_json['output']['coefficients_table']['coefficients']})

    print(coefficients)

    return None

def compare_sklearn() -> None:
    # X, y = datasets.make_friedman1(n_samples=10000, n_features=5)
    X, y, coef = datasets.make_regression(n_samples=100, n_features=10, n_informative=10, noise=10, coef=True)
    data = pd.DataFrame(np.concatenate((X, y.reshape(len(y),1)), axis=1), 
                            columns=[*[f'feat_{i+1}' for i in range(10)], 'y'])
    glm: GLE = GLE(solver='IRLSM',
                    family= "gaussian",
                    lambda_ = 0,
                    compute_p_values = True,
                    standardize = True,
                    intercept = True,
                    link = 'Identity')

    glm.train(x = list(data.columns.values[:-1]), 
            y = data.columns.values[-1], 
            training_frame = H2OFrame(data))

    print(glm.coef())
    print(coef)

    return None

if __name__ == '__main__':
    h2o.init()

    compare_poisson()
    # compare_sklearn()

    H2OCluster().shutdown()
