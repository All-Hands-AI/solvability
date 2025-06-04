import pytest

from solvability.models.featurizer import Feature


def test_feature_to_tool_description_field():
    """Test to_tool_description_field property."""
    feature = Feature(identifier="test", description="Test description")
    field = feature.to_tool_description_field

    # There's not much structure here, but we can check the expected type and make
    # sure the other fields are propagated.
    assert field["type"] == "boolean"
    assert field["description"] == "Test description"


def test_feature_embedding_dimensions(feature_embedding):
    """Test dimensions property."""
    dimensions = feature_embedding.dimensions
    assert isinstance(dimensions, list)
    assert set(dimensions) == {"feature1", "feature2", "feature3"}


def test_feature_embedding_coefficients(feature_embedding):
    """Test coefficient method."""
    # These values are manually computed from the results in the fixture's samples.
    assert feature_embedding.coefficient("feature1") == 0.5
    assert feature_embedding.coefficient("feature2") == 0.5
    assert feature_embedding.coefficient("feature3") == 1.0

    # Non-existent features should not have a coefficient.
    assert feature_embedding.coefficient("non_existent") is None


def test_featurizer_system_message(featurizer):
    """Test system_message method."""
    message = featurizer.system_message()
    assert message["role"] == "system"
    assert message["content"] == "Test system prompt"


def test_featurizer_user_message(featurizer):
    """Test user_message method."""
    # With cache
    message = featurizer.user_message("Test issue", set_cache=True)
    assert message["role"] == "user"
    assert message["content"] == "Test message prefix: Test issue"
    assert "cache_control" in message
    assert message["cache_control"]["type"] == "ephemeral"

    # Without cache
    message = featurizer.user_message("Test issue", set_cache=False)
    assert message["role"] == "user"
    assert message["content"] == "Test message prefix: Test issue"
    assert "cache_control" not in message


def test_featurizer_tool_choice(featurizer):
    """Test tool_choice property."""
    tool_choice = featurizer.tool_choice
    assert tool_choice["type"] == "function"
    assert tool_choice["function"]["name"] == "call_featurizer"


def test_featurizer_tool_description(featurizer):
    """Test tool_description property."""
    tool_desc = featurizer.tool_description
    assert tool_desc["type"] == "function"
    assert tool_desc["function"]["name"] == "call_featurizer"
    assert "description" in tool_desc["function"]

    # Check that all features are included in the properties
    properties = tool_desc["function"]["parameters"]["properties"]
    for feature in featurizer.features:
        assert feature.identifier in properties
        assert properties[feature.identifier]["type"] == "boolean"
        assert properties[feature.identifier]["description"] == feature.description


@pytest.mark.parametrize("samples", [1, 10, 100])
def test_featurizer_embed(samples, featurizer):
    """Test the embed method to ensure it generates the right number of samples and computes the metadata correctly."""
    embedding = featurizer.embed("Test issue", samples=samples)

    # We should get the right number of samples.
    assert len(embedding.samples) == samples

    # Because of the mocks, all the samples should be the same (and be correct).
    assert all(sample == embedding.samples[0] for sample in embedding.samples)
    assert embedding.samples[0]["feature1"] is True
    assert embedding.samples[0]["feature2"] is False
    assert embedding.samples[0]["feature3"] is True

    # And all the metadata should be correct (we know the token counts because
    # they're mocked, so just count once per sample).
    assert embedding.prompt_tokens == 10 * samples
    assert embedding.completion_tokens == 5 * samples

    # These timings are real, so best we can do is check that they're positive.
    assert embedding.response_latency > 0.0


@pytest.mark.parametrize("samples", [1, 10, 100])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_featurizer_embed_batch(samples, batch_size, featurizer):
    """Test the embed_batch method to ensure it correctly handles all issues in the batch."""
    embeddings = featurizer.embed_batch([f"Issue {i}" for i in range(batch_size)], samples=samples)

    # Make sure that we get an embedding for each issue.
    assert len(embeddings) == batch_size

    # Since the embeddings are computed from a mocked completionc all, they should
    # all be the same. We can check that they're well-formatted by applying the same
    # checks as in `test_featurizer_embed`.
    for embedding in embeddings:
        assert all(sample == embedding.samples[0] for sample in embedding.samples)
        assert embedding.samples[0]["feature1"] is True
        assert embedding.samples[0]["feature2"] is False
        assert embedding.samples[0]["feature3"] is True

        assert len(embedding.samples) == samples
        assert embedding.prompt_tokens == 10 * samples
        assert embedding.completion_tokens == 5 * samples
        assert embedding.response_latency > 0.0
