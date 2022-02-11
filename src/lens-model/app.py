from typing import List, Dict
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from patsy.highlevel import dmatrices
from sklearn.metrics import explained_variance_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from glm_estimator import (
    GLMEstimatorSummary,
    FeatureSummary,
    FeatureSummaryData,
    GLMBasicInfo,
    GLMSummaryPayload,
)


def _get_feature_summary_data(
    df: DataFrame, info: GLMBasicInfo, feature: str
) -> FeatureSummaryData:
    # discretize into 50 bins
    bins = pd.cut(df[feature], bins=50)

    # for visibility
    target: str = info["target"]
    prediction: str = info["prediction"]
    weight: str = info["weight"]

    # replace data with sums over feature categories
    df = df.groupby(bins).agg(
        **{
            "sum_target": (target, "sum"),
            "sum_prediction": (prediction, "sum"),
            "sum_weight": (weight, "sum"),
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


def summarize(
    df: DataFrame,
    info: GLMBasicInfo,
) -> GLMEstimatorSummary:

    ## IDENTIFIERS
    desc_payload = {
        "name": info["name"],
        "desc": info["formula"],
        "target": info["target"],
        "prediction": info["prediction"],
    }

    ## MODEL ERROR STRUCTURE
    error_payload = {
        "var_weights": info["weight"],
        "link_function": "identity",
        "error_dist": "gaussian",
    }

    ## SCORING

    # explained variance
    expv = explained_variance_score(df[info["target"]], df[info["prediction"]])

    # other scores (TBD) ...

    # combined scores payload
    scores_payload = {"explained_variance": expv}

    ## FEATURE SUMMARIES
    feat_summaries_payload: Dict[str, List[FeatureSummary]] = {
        "feature_summary": _feature_summaries(df, info)
    }

    # mypy, why u mad bro? seems like false positive on the typing error
    summary_payload: GLMSummaryPayload = {
        **desc_payload,  # type: ignore
        **error_payload,
        **scores_payload,
        **feat_summaries_payload,
    }

    # return payload associated with the data
    return GLMEstimatorSummary(**summary_payload)
