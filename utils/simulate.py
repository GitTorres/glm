import pandas as pd
from pandas import DataFrame
from numpy.random import Generator, PCG64
import numpy as np
from typing import List, Optional

class Simulate():
    def __init__(self, dist: str) -> None:
        error_dist = 'use "poisson", "gamma" or "tweedie" as values for the type parameter'
        assert dist in ['poisson', 'gamma', 'tweedie'], error_dist

        self.dist: Optional[str] = dist
        self.data: Optional[DataFrame] = None
        self.coeffs: Optional[List] = None
        self.response: Optional[str] = None
        self.predictors: Optional[List] = None
        self.model: Optional[str] = None

    def _has_coeffs(self) -> bool:
        if self.coeffs:
            return True
        else:
            return False

    def clear(self) -> None:
        print('erasing existing simulated data')
        self.data = None
        self.coeffs = None
        self.response = None
        self.predictors = None
        self.model = None

    
    def generate_model_data(self, n: int) -> None:

        assert n >= 0, 'use a positive number as the value for the n parameter'
        assert self.coeffs is None, 'please clear results of existing simulation before running a new one'
        
        self.generate_poisson_model_data(n)

        return None

    def generate_poisson_model_data(self, n: int) -> None:
        """
        This function generates a dataframe of predictors and a Poisson response.

        :::Function Parameters:::
        n - the size of the random data sample

        :::Predictors:::
        X1 ~ Uniform[a,b]
        X2 - Normal(u, s) / 10
        X3 - Discrete Uniform {0,1,2}
        X4 = X1*X2 / 10
        X5 = X2*X3 / 10
        X = [1 | X1 | X2 | X3 | X4 | X5]

        :::True Response Coefficients:::
        B = [B0, B1,..., B5], where Bi ~ Uniform[-.8,.8]
        
        :::Response Distribution:::
        Y ~ Poisson(lambda)
        E[Y|X] == Var[Y|X] == lambda
    
        :::Model for Response:::
        ln(E[Y|X]) = X.matmul(B)
        Y ~ Poisson(lambda=exp(X.matmul(B)))
        """

        rng = Generator(PCG64(12345))

        #parameters
        coeffs = rng.uniform(low=-2, high=2, size=(6))

        #predictors
        X0 = np.ones(shape=(n,1))
        X1 = rng.uniform(low=-2, high=2, size=(n,1))
        X2 = rng.uniform(low=-2, high=2, size=(n,1))
        X3 = rng.normal(loc=0, scale=1, size=(n,1))
        X4 = rng.normal(loc=0, scale=1, size=(n,1))
        X5 = rng.normal(loc=0, scale=1, size=(n,1))

        included_columns = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'Y']
        included_predictors = [X0, X1, X2, X3, X4, X5]
        included_coeffs = coeffs[0:len(included_predictors)]

        X = np.concatenate(included_predictors, axis=1)
        conditional_mean = np.exp((np.matmul(X, included_coeffs))).round(1)

        Y = rng.poisson(lam=conditional_mean).reshape(n, 1)
        data = np.concatenate((X, Y), axis=1)

        self.data = pd.DataFrame(data, columns=included_columns)
        self.response = included_columns.pop()
        self.coeffs = included_coeffs.ravel()
        # self.model = f'Y = exp({coeffs[0][0]} + ' + '+ '.join([f'{coeff}*X_{i}' for (i, coeff) in enumerate(included_coeffs.ravel()) if i > 0]) + ')'

        return None
