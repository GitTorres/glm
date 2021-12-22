"""
Module Purpose:
Create model summary objects to save to Mongodb for reproducibility
"""

import abc
from typing import List, Dict, TypedDict
from urllib.parse import quote_plus
import time
from uuid import uuid4
from pymongo import MongoClient


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


class GLMEstimatorSummary(SupervisedEstimatorSummary):
    """GLM Estimator"""

    def __init__(
        self,
        name: str,
        desc: str,
        target: str,
        prediction: str,
        var_weights: str,
        link_function: str,
        error_dist: str,
        explained_variance: float,
        feature_summary: List[FeatureSummary],
    ) -> None:
        super().__init__(name, desc, target, prediction)
        self.var_weights: str = var_weights
        self.link_function: str = link_function
        self.error_dist: str = error_dist
        self.feature_summary = feature_summary
        self.explained_variance = explained_variance

    def _save_model_summary_to_db(
        self, client: MongoClient, db: str, collection: str
    ) -> Dict:

        # get db
        mydb = client[db]

        # get collection
        clcn = mydb[collection]

        # add model summary to collection
        created_at = time.time()
        info = clcn.insert_one(
            {"_id": uuid4(), "created_time": created_at, **self.show()}
        )

        return {
            "name": self.name,
            "inserted_id": info.inserted_id,
            "created_time": created_at,
        }

    def show(self) -> Dict:
        """View summary of estimator statistics"""

        return {k: v for k, v in self.__dict__.items() if "__" not in k}

    def save(self) -> None:
        assert all(self.__dict__.values()), "Set all properties before saving"

        # get client
        username = quote_plus("root")
        password = quote_plus("OTNmYTdjYmZkMjE5ZmYzODg0MDZiYWJh")
        uri = f"mongodb://{username}:{password}@localhost:27017/?uuidRepresentation=standard"
        client = MongoClient(uri)

        # save model to Mongo db
        response = self._save_model_summary_to_db(
            client, db="models", collection="models"
        )

        print(response)
