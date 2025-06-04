import importlib
from pathlib import Path

import click


@click.group()
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    envvar="SOLVABILITY_MODEL_PATH",
    help="Path to the trained classifier.",
)
@click.pass_context
def cli(ctx: click.Context, model_path: Path | None):
    """
    Command line interface for the OpenHands solvability classifier.
    """
    if model_path is None:
        raise click.BadParameter("Model path is required. Please provide a valid path to a solvability classifier.")
    ctx.obj = model_path


# Auto-load other modules in this directory.
cli_dir = Path(__file__).parent
for file in cli_dir.glob("*.py"):
    module_name = file.stem
    if module_name in ("cli", "__init__"):
        continue
    importlib.import_module(f"solvability.cli.{module_name}")
