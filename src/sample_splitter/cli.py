import tomllib
from importlib import resources
from pathlib import Path

import soundfile as sf
import typer

from sample_splitter import audio_io

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
def scan(input_path: Path) -> None:
    """Print a per-file inventory report for every audio file in a folder."""
    if not input_path.is_dir():
        typer.echo(f"Error: {input_path} is not a directory", err=True)
        raise typer.Exit(code=1)

    scanned = 0
    skipped = []

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            info = audio_io.probe(file_path)
        except sf.LibsndfileError:
            skipped.append(file_path.name)
            continue

        scanned += 1
        bit_depth = f"{info.bit_depth}-bit" if info.bit_depth is not None else "unknown bit depth"
        typer.echo(
            f"{file_path.name}: {info.format}, {info.sample_rate} Hz, "
            f"{bit_depth}, {info.channels} ch, {info.duration_s:.2f}s"
        )

    typer.echo(f"\n{scanned} file(s) scanned, {len(skipped)} skipped")
    if skipped:
        typer.echo(f"Skipped (not recognised as audio): {', '.join(skipped)}")


@app.command()
def split(ctx: typer.Context, input_path: Path) -> None:
    """Extract detected samples from splittable tracks into a JSON manifest."""
    _run_stub(ctx, input_path)


@app.command()
def name(ctx: typer.Context, input_path: Path) -> None:
    """Classify extracted samples and file them into the taxonomy tree."""
    _run_stub(ctx, input_path)
