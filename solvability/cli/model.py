from pathlib import Path

import click
import rich

from solvability.cli.cli import cli
from solvability.models import SolvabilityClassifier


@cli.group()
def model():
    """Commands related to the model."""
    pass


@model.group()
def features():
    """Commands related to the features in the classifier."""
    pass


@features.command()
@click.pass_obj
def list(model_path: Path):
    """
    List the features in a trained classifier.

    Args:
        classifier_path (str): Path to the trained classifier.
    """
    with model_path.open("r") as f:
        classifier = SolvabilityClassifier.model_validate_json(f.read())

    for feature in classifier.featurizer.features:
        rich.print(feature.identifier)


@features.command()
@click.argument("feature_name")
@click.pass_obj
def description(model_path: Path, feature_name: str):
    """
    Get the description of a specific feature in a trained classifier.
    """
    with model_path.open("r") as f:
        classifier = SolvabilityClassifier.model_validate_json(f.read())

    for feature in classifier.featurizer.features:
        if feature.identifier == feature_name:
            rich.print(feature.description)
            return

    rich.print(f"Feature '{feature_name}' not found.")
    click.exit(1)
