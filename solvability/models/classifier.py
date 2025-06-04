from __future__ import annotations

import base64
import pickle
from typing import Any

import numpy as np
import pandas as pd
import shap
from pydantic import (
    BaseModel,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from solvability.models.featurizer import Feature, Featurizer
from solvability.models.importance_strategy import ImportanceStrategy
from solvability.models.report import SolvabilityReport


class SolvabilityClassifier(BaseModel):
    """
    Solvability classifier that chains a featurizer and a scikit-learn classifier.
    """

    identifier: str
    """
    The identifier for the classifier.
    """

    featurizer: Featurizer
    """
    The featurizer to use for transforming the input data.
    """

    classifier: RandomForestClassifier
    """
    The classifier to use for predicting the solvability of the input data.
    """

    importance_strategy: ImportanceStrategy = ImportanceStrategy.IMPURITY
    """
    Strategy to use for calculating feature importance.
    """

    samples: int = 10
    """
    Number of samples to use for calculating feature embedding coefficients.
    """

    random_state: int | None = None
    """
    Random state for reproducibility.
    """

    _classifier_attrs: dict[str, Any] = PrivateAttr(default_factory=dict)
    """
    Private dictionary to store the results of secondary classifier calculations. Use the `attr_` properties to access
    the values.

    This field is never serialized, so the values will not persist when using the standard model dumping/loading
    methods.
    """

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @model_validator(mode="after")
    def validate_random_state(self) -> SolvabilityClassifier:
        """
        Validate the random state configuration between this object and the classifier.
        """
        # If both random states are set, they definitely need to agree.
        if self.random_state is not None and self.classifier.random_state is not None:
            if self.random_state != self.classifier.random_state:
                raise ValueError("The random state of the classifier and the top-level classifier must agree.")

        # Otherwise, we'll always set the classifier's random state to the top-level one.
        self.classifier.random_state = self.random_state

        return self

    @property
    def features_(self) -> pd.DataFrame:
        """
        Get the features used by the classifier for the most recent inputs.
        """
        if "features_" not in self._classifier_attrs:
            raise ValueError("SolvabilityClassifier.transform() has not yet been called.")
        return self._classifier_attrs["features_"]

    @property
    def cost_(self) -> pd.DataFrame:
        """
        Get the cost of the classifier for the most recent inputs.
        """
        if "cost_" not in self._classifier_attrs:
            raise ValueError("SolvabilityClassifier.transform() has not yet been called.")
        return self._classifier_attrs["cost_"]

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Get the feature importances for the most recent inputs.
        """
        if "feature_importances_" not in self._classifier_attrs:
            raise ValueError(
                "No SolvabilityClassifier methods that produce feature importances (.fit(), .predict_proba(), and "
                ".predict()) have been called."
            )
        return self._classifier_attrs["feature_importances_"]

    @property
    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.
        """
        try:
            check_is_fitted(self.classifier)
            return True
        except NotFittedError:
            return False

    def transform(self, issues: pd.Series) -> pd.DataFrame:
        """
        Transform the input issues using the featurizer.

        Args:
            issues: A pandas Series containing the issue descriptions.

        Returns:
            pd.DataFrame: A DataFrame containing the transformed features.
        """
        feature_embeddings = self.featurizer.embed_batch(issues, samples=self.samples)
        df = pd.DataFrame(embedding.to_row() for embedding in feature_embeddings)

        # Split into two sets of columns -- those that correspond to a featre and
        # those that don't.
        feature_columns = [feature.identifier for feature in self.featurizer.features]
        cost_columns = [col for col in df.columns if col not in feature_columns]

        self._classifier_attrs["features_"] = df[feature_columns]
        self._classifier_attrs["cost_"] = df[cost_columns]

        return self.features_

    def fit(self, issues: pd.Series, labels: pd.Series) -> SolvabilityClassifier:
        """
        Fit the classifier to the input issues and labels.

        Args:
            issues: A pandas Series containing the issue descriptions.

            labels: A pandas Series containing the labels (0 or 1) for each issue.

        Returns:
            SolvabilityClassifier: The fitted classifier.
        """
        features = self.transform(issues)
        self.classifier.fit(features, labels)

        self._classifier_attrs["feature_importances_"] = self._importance(
            features, self.classifier.predict_proba(features)
        )

        return self

    def predict_proba(self, issues: pd.Series) -> np.ndarray:
        """
        Predict the solvability of the input issues by returning the probabilities that the issues are solvable.
        """
        features = self.transform(issues)
        scores = self.classifier.predict_proba(features)

        self._classifier_attrs["feature_importances_"] = self._importance(features, scores)

        return scores

    def predict(self, issues: pd.Series) -> np.ndarray:
        """
        Predict the solvability of the input issues by returning the labels (0 or 1).
        """
        probabilities = self.predict_proba(issues)
        labels = probabilities[:, 1] >= 0.5
        return labels

    def _importance(self, features: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """
        Calculate the relative importance of features in producing the scores.
        """
        match self.importance_strategy:
            case ImportanceStrategy.SHAP:
                explainer = shap.TreeExplainer(self.classifier)
                shap_values = explainer.shap_values(features)
                return shap_values.mean(axis=0)[:, 1]

            case ImportanceStrategy.PERMUTATION:
                result = permutation_importance(
                    self.classifier,
                    features,
                    scores[:, 1],
                    n_repeats=10,
                    random_state=self.random_state,
                )
                return result.importances_mean

            case ImportanceStrategy.IMPURITY:
                return self.classifier.feature_importances_

    def add_features(self, features: list[Feature]) -> SolvabilityClassifier:
        """
        Add the specified features to the classifier.
        """
        for feature in features:
            if feature not in self.featurizer.features:
                self.featurizer.features.append(feature)
        return self

    def forget_features(self, features: list[Feature]) -> SolvabilityClassifier:
        """
        Forget the specified features from the classifier.
        """
        for feature in features:
            try:
                self.featurizer.features.remove(feature)
            except ValueError:
                continue
        return self

    @field_serializer("classifier")
    @staticmethod
    def _rfc_to_json(rfc: RandomForestClassifier) -> str:
        """
        Convert a RandomForestClassifier to a JSON-compatible value (a string).
        """
        return base64.b64encode(pickle.dumps(rfc)).decode("utf-8")

    @field_validator("classifier", mode="before")
    @staticmethod
    def _json_to_rfc(value: str | RandomForestClassifier) -> RandomForestClassifier:
        """
        Convert a JSON-compatible value (a string) back to a RandomForestClassifier.
        """
        if isinstance(value, RandomForestClassifier):
            return value

        if isinstance(value, str):
            try:
                model = pickle.loads(base64.b64decode(value))
                if isinstance(model, RandomForestClassifier):
                    return model
            except Exception as e:
                raise ValueError(f"Failed to decode the classifier: {e}")

        raise ValueError("The classifier must be a RandomForestClassifier or a JSON-compatible dictionary.")

    def solvability_report(self, issue: str, **kwargs: Any) -> SolvabilityReport:
        """
        Generate a solvability report for the given issue.

        Args:
            issue: The issue description for which to generate the report.
            kwargs: Additional metadata to include in the report.

        Returns:
            SolvabilityReport: The generated solvability report.
        """
        if not self.is_fitted:
            raise ValueError("The classifier must be fitted before generating a report.")

        scores = self.predict_proba(pd.Series([issue]))

        return SolvabilityReport(
            identifier=self.identifier,
            issue=issue,
            score=scores[0, 1],
            features=self.features_.iloc[0].to_dict(),
            samples=self.samples,
            importance_strategy=self.importance_strategy,
            # Unlike the features, the importances are just a series with no link
            # to the actual feature names. For that we have to recombine with the
            # feature identifiers.
            feature_importances=dict(
                zip(
                    self.featurizer.feature_identifiers(),
                    self.feature_importances_.tolist(),
                )
            ),
            random_state=self.random_state,
            metadata=dict(kwargs) if kwargs else None,
            # Both cost and response_latency are columns in the cost_ DataFrame,
            # so we can get both by just unpacking the first row.
            **self.cost_.iloc[0].to_dict(),
        )

    def __call__(self, issue: str, **kwargs: Any) -> SolvabilityReport:
        """
        Generate a solvability report for the given issue.
        """
        return self.solvability_report(issue, **kwargs)
