import lens
from patsy.highlevel import dmatrices
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from lens import GLMBasicInfo, GLMEstimatorSummary
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas_profiling import ProfileReport


def create_and_upload_breast_cancer_model(data: DataFrame) -> None:

    # get design matrix
    data["weight"] = 1
    features = ["mean radius", "area error"]
    wilk_formula = "Q('worst concave points') ~ 0 + Q('mean radius') + Q('area error')"
    y, D = dmatrices(wilk_formula, data=data)

    # fit model
    linreg = LinearRegression(fit_intercept=False).fit(
        X=D,
        y=y.ravel(),
    )

    # attach predictions to our data
    data["predicted"] = linreg.predict(D)
    del D, y

    model_info: GLMBasicInfo = {
        "formula": wilk_formula,
        "features": features,
        "name": "jose_bc_cm3",
        "prediction": "predicted",
        "target": "worst concave points",
        "weight": "weight",
    }

    # create a model summary
    doc: GLMEstimatorSummary = lens.summarize(df=data, info=model_info)

    # save summary document to database
    doc.save()

def save_profile(data: DataFrame) -> None:
    profile = ProfileReport(data, title="Pandas Profiling Report", interactions=None)
    profile.to_file("your_report.html")



if __name__ == "__main__":
    data = datasets.load_breast_cancer(as_frame=True).data

    # save_profile(data)
    create_and_upload_breast_cancer_model(data)
