import tomllib
from importlib import resources
from pathlib import Path

import typer

app = typer.Typer()


def _load_default_config() -> dict:
    default_config_path = resources.files("sample_splitter.config") / "default.toml"
    with default_config_path.open("rb") as f:
        return tomllib.load(f)


def _run_stub(ctx: typer.Context, input_path: Path) -> None:
    config = _load_default_config()
    typer.echo(f"{ctx.command.name}: {input_path}")
    typer.echo(config)


@app.command()
def scan(ctx: typer.Context, input_path: Path) -> None:
    """Classify tracks as splittable or montage and report gap statistics."""
    _run_stub(ctx, input_path)


@app.command()
def split(ctx: typer.Context, input_path: Path) -> None:
    """Extract detected samples from splittable tracks into a JSON manifest."""
    _run_stub(ctx, input_path)


@app.command()
def name(ctx: typer.Context, input_path: Path) -> None:
    """Classify extracted samples and file them into the taxonomy tree."""
    _run_stub(ctx, input_path)
