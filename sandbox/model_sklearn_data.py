from typing import List, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from patsy.highlevel import dmatrices
from sklearn.metrics import explained_variance_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# from src.model_lens.app import summarize, GLMBasicInfo, GLMEstimatorSummary
from lens_model import summarize, GLMBasicInfo, GLMEstimatorSummary


def test():
    data = datasets.load_breast_cancer(as_frame=True).data

    # get design matrix
    data["weight"] = 1
    features = ["mean radius", "area error"]
    wilk_formula = "Q('worst concave points') ~ 1 + Q('mean radius') + Q('area error')"
    y, D = dmatrices(wilk_formula, data=data)

    # fit model
    linreg = LinearRegression(fit_intercept=True).fit(
        X=D,
        y=y.ravel(),
    )

    # attach predictions to our data
    data["predicted"] = linreg.predict(D)
    del D, y

    model_info: GLMBasicInfo = {
        "formula": wilk_formula,
        "features": features,
        "name": "breast cancer model 1",
        "prediction": "predicted",
        "target": "worst concave points",
        "weight": "weight",
    }

    doc: GLMEstimatorSummary = summarize(df=data, info=model_info)

    # print(doc.created_time)

    # save summary document to database
    doc.save()


if __name__ == "__main__":
    test()
