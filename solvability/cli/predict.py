from pathlib import Path

import click
import requests
import rich

from solvability.cli.cli import cli
from solvability.models import SolvabilityClassifier
from solvability.models.config import LLMConfig


@cli.command()
@click.argument("issue", type=str)
@click.option(
    "--output-type",
    type=click.Choice(["simple", "detailed", "summary"]),
    default="simple",
    help="Output type for the prediction.",
)
@click.option("--threshold", type=float, default=0.5, help="Threshold for solvability.")
@click.pass_obj
def predict(model_path: Path, issue: str, output_type: str, threshold: float):
    """
    Predict the solvability of a given issue using a trained classifier.
    """
    with model_path.open("r") as f:
        classifier = SolvabilityClassifier.model_validate_json(f.read())

    report = classifier.solvability_report(issue)

    match output_type:
        case "simple":
            label = report.score >= threshold
            rich.print(label)
        case "detailed":
            rich.print(report)
        case "summary":
            summary = report.summarize(llm_config=LLMConfig.from_dotenv())
            rich.print(summary)


@cli.command()
@click.argument("owner", type=str)
@click.argument("repo", type=str)
@click.argument("issue_number", type=int)
@click.option(
    "--output-type",
    type=click.Choice(["simple", "detailed", "summary"]),
    default="simple",
    help="Output type for the prediction.",
)
@click.option("--threshold", type=float, default=0.5, help="Threshold for solvability.")
@click.option("--samples", type=int, help="Number of samples to use for prediction.")
@click.pass_obj
def predict_gh(
    model_path: Path, owner: str, repo: str, issue_number: int, output_type: str, threshold: float, samples: int | None
):
    """
    Predict the solvability of a GitHub issue using a trained classifier.
    """
    with model_path.open("r") as f:
        classifier = SolvabilityClassifier.model_validate_json(f.read())

    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}",
    )
    issue_deescription = response.json()["body"]

    if samples is not None:
        classifier.samples = samples

    report = classifier.solvability_report(
        issue_deescription,
        owner=owner,
        repo=repo,
        issue_number=issue_number,
    )
    match output_type:
        case "simple":
            label = report.score >= threshold
            rich.print(label)
        case "detailed":
            rich.print(report)
        case "summary":
            summary = report.summarize(llm_config=LLMConfig.from_dotenv())
            rich.print(summary)
