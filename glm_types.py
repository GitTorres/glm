"""
Module Purpose:
Create model summary objects to save to Mongodb for reproducibility
"""
from dataclasses import dataclass
import abc
from typing import List, Dict, TypedDict
import requests


class GLMBasicInfo(TypedDict):  # pylint: disable=missing-class-docstring
    name: str
    formula: str
    features: List[str]
    prediction: str
    target: str


class FeatureSummaryData(TypedDict):  # pylint: disable=missing-class-docstring
    bin_edge_right: List[float]
    sum_target: List[float]
    sum_prediction: List[float]
    sum_weight: List[float]
    wtd_avg_prediction: List[float]
    wtd_avg_target: List[float]


class FeatureSummary(TypedDict):  # pylint: disable=missing-class-docstring
    name: str
    data: FeatureSummaryData


class GLMSummaryPayload(TypedDict):  # pylint: disable=missing-class-docstring
    name: str
    desc: str
    target: str
    prediction: str
    var_weights: str
    link_function: str
    error_dist: str
    explained_variance: float
    feature_summary: List[FeatureSummary]


class SupervisedEstimatorSummary(metaclass=abc.ABCMeta):
    """Abstract class which forces Estimator Summaries to define a save() and show() method"""

    def __init__(
        self,
        name: str,
        desc: str,
        target: str,
        prediction: str,
    ) -> None:
        self.name: str = name
        self.desc: str = desc
        self.target: str = target
        self.prediction: str = prediction

    @abc.abstractmethod
    def show(self) -> Dict:
        """View summary of estimator statistics"""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self) -> None:
        """Save summary of estimator statistics"""
        raise NotImplementedError


@dataclass
class GLMEstimatorSummary(SupervisedEstimatorSummary):
    """
    Regression model summaries.
    """

    name: str
    desc: str
    target: str
    prediction: str
    var_weights: str
    link_function: str
    error_dist: str
    explained_variance: float
    feature_summary: List[FeatureSummary]

    def show(self) -> Dict:
        """View summary of estimator statistics"""

        return {k: v for k, v in self.__dict__.items() if "__" not in k}

    def save(self) -> None:
        """Save model summary to db"""

        url = "http://localhost:8000/modelsummary/regression"
        assert all(self.__dict__.values()), "Set all properties before saving"

        response = requests.put(url, json=self.show())

        if response.text != "error":
            print(f"model summary saved at {response.text}")
        else:
            print("error saving model summary")
