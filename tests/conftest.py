"""
Shared fixtures for all tests.
"""
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from solvability.models.classifier import SolvabilityClassifier
from solvability.models.featurizer import Feature, FeatureEmbedding, Featurizer


@pytest.fixture
def features() -> list[Feature]:
    """Create a list of features for testing."""
    return [
        Feature(identifier="feature1", description="Test feature 1"),
        Feature(identifier="feature2", description="Test feature 2"),
        Feature(identifier="feature3", description="Test feature 3")
    ]


@pytest.fixture
def feature_embedding() -> FeatureEmbedding:
    """Create a feature embedding for testing."""
    return FeatureEmbedding(
        samples=[
            {"feature1": True, "feature2": False, "feature3": True},
            {"feature1": False, "feature2": True, "feature3": True}
        ],
        prompt_tokens=10,
        completion_tokens=5,
        response_latency=0.1
    )


@pytest.fixture
def featurizer(mock_completion_response, features) -> Featurizer:
    """
    Create a featurizer for testing.

    Mocks out any calls to `completion`
    """
    pytest.MonkeyPatch().setattr(
        "solvability.models.featurizer.completion",
        lambda *args, **kwargs: mock_completion_response
    )

    featurizer = Featurizer(
        system_prompt="Test system prompt",
        message_prefix="Test message prefix: ",
        features=features
    )

    return featurizer

@pytest.fixture
def mock_completion_response() -> dict[str, Any]:
    """Create a mock response for the feature sample model."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"feature1": true, "feature2": false, "feature3": true}'
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    return mock_response


@pytest.fixture
def mock_classifier():
    """Create a mock classifier for testing."""
    clf = RandomForestClassifier(random_state=42)
    # Initialize with some dummy data to avoid errors
    X = np.array([[0, 0, 0], [1, 1, 1]])
    y = np.array([0, 1])
    clf.fit(X, y)
    return clf


@pytest.fixture
def solvability_classifier(featurizer, mock_classifier):
    """Create a SolvabilityClassifier instance for testing."""
    return SolvabilityClassifier(
        identifier="test-classifier",
        featurizer=featurizer,
        classifier=mock_classifier,
        random_state=42
    )