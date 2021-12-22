import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from patsy.highlevel import dmatrices
from glm_types import (
    GLMEstimatorSummary,
    FeatureSummary,
    FeatureSummaryData,
    GLMBasicInfo,
    GLMSummaryPayload,
)
from typing import List, Dict
from sklearn.metrics import explained_variance_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression

NUM_DATASETS = 10
OUT_PATH = "./regression_data"
PREDICTION_COL_NAME = "prediction"
TARGET_COL_NAME = "y"


def _get_feature_summary_data(
    df: DataFrame, info: GLMBasicInfo, feature: str
) -> FeatureSummaryData:
    # discretize into 50 bins
    bins = pd.cut(df[feature], bins=50)

    # for visibility
    target: str = info["target"]
    prediction: str = info["prediction"]

    # replace data with sums over feature categories
    df = df.groupby(bins).agg(
        **{
            "sum_target": (target, "sum"),
            "sum_prediction": (prediction, "sum"),
            "sum_weight": ("weight", "sum"),
        }
    )

    # assign averages
    df = df.assign(
        bin_edge_right=df.index.categories.right,
        wtd_avg_prediction=df["sum_prediction"] / df["sum_weight"],
        wtd_avg_target=df["sum_target"] / df["sum_weight"],
    )

    # drop categories with no observations
    df = df[df.sum_weight > 0]

    response: FeatureSummaryData = df.to_dict("list")

    return response


def _feature_summaries(df: DataFrame, info: GLMBasicInfo) -> List[FeatureSummary]:
    """Return a list of feature summaries for a given model"""

    return [
        {"name": feature, "data": _get_feature_summary_data(df, info, feature)}
        for feature in info["features"]
    ]


def gen_data() -> DataFrame:
    """
    Generate randomized regression data set.

    The true models are of the form Y|X = dot(X, B_true) + N(0, `noise`).

    ==Response==
    DataFrame
        - The generated data set
    """

    # generate regression data set
    X, y, coef = datasets.make_regression(
        n_samples=1000000, n_features=15, n_informative=5, noise=5, coef=True
    )

    _data = pd.DataFrame(
        np.append(X, y.reshape(y.shape[0], -1), axis=1),
        columns=[*[f"X{i}" for i in range(X.shape[1])], "y"],
    )

    # add record weights (use 1.0 for ordinary least squares data)
    _data["weight"] = 1

    return _data


def get_initial_features(df: DataFrame, target: str) -> List[str]:
    """Return the formula representing best initial model for the input data set"""

    # features are variables with over 25% correlation with y
    features = list(df.corr().loc[target].pipe(lambda s: s[s > 0.25]).index)

    # remove 'y' from the end of features
    features.pop()

    # return formula (`target ~ linear predictor`)
    return features


def fit_predict(df: DataFrame, formula: str) -> Series:
    """For each data set, attach the OLS predictions and return a DataFrame with model formulas"""

    # get design matrix
    y, D = dmatrices(formula, data=df)

    # fit model
    linreg = LinearRegression(fit_intercept=False).fit(
        X=D,
        y=y.ravel(),
    )

    # attach predictions to our data
    predicted = linreg.predict(D)
    del D, y

    return predicted


def summarize(
    df: DataFrame,
    info: GLMBasicInfo,
) -> GLMEstimatorSummary:

    # store payload in object

    # identifiers
    desc_payload = {
        "name": info["name"],
        "desc": info["formula"],
        "target": info["target"],
        "prediction": info["prediction"],
    }

    # model error structure
    error_payload = {
        "var_weights": "weight",
        "link_function": "identity",
        "error_dist": "gaussian",
    }

    # scoring
    scores_payload = {
        "explained_variance": explained_variance_score(
            df[info["target"]], df[info["prediction"]]
        )
    }

    # feature summaries
    summaries_payload: Dict[str, List[FeatureSummary]] = {
        "feature_summary": _feature_summaries(df, info)
    }

    # mypy, why u mad bro? seems like false positive on the typing error
    summary_payload: GLMSummaryPayload = {
        **desc_payload,  # type: ignore
        **error_payload,
        **scores_payload,
        **summaries_payload,
    }

    # return payload associated with the data
    return GLMEstimatorSummary(**summary_payload)


def main() -> None:
    # generate modeling data
    for i in range(0, NUM_DATASETS):
        # simulate regression data (sklearn used for underlying pseudorandom numbers)
        data: DataFrame = gen_data().rename({"y": TARGET_COL_NAME})

        # get initial features for linear model (via pairwise correlation)
        features: List[str] = get_initial_features(data, TARGET_COL_NAME)

        # define Wilkinson formula
        wilk_formula = f"{TARGET_COL_NAME} ~ 0 + {'+'.join(features)}"

        # attach the predictions for the selected formula
        data[PREDICTION_COL_NAME] = fit_predict(data, wilk_formula)

        # basic model info
        model_info: GLMBasicInfo = {
            "formula": wilk_formula,
            "features": features,
            "name": f"model_{i}",
            "prediction": PREDICTION_COL_NAME,
            "target": TARGET_COL_NAME,
        }

        # get summary of model (the juice right here)
        doc: GLMEstimatorSummary = summarize(df=data, info=model_info)

        # save summary document to database
        doc.save()


if __name__ == "__main__":
    main()
