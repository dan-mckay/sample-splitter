import statistics
import tomllib
from importlib import resources
from pathlib import Path

import soundfile as sf
import typer

from sample_splitter import analysis, audio_io, manifest

app = typer.Typer()


def _load_default_config() -> dict:
    default_config_path = resources.files("sample_splitter.config") / "default.toml"
    with default_config_path.open("rb") as f:
        return tomllib.load(f)


def _build_analysis_config(config: dict) -> analysis.AnalysisConfig:
    splitter = config["splitter"]
    analysis_settings = config["analysis"]
    return analysis.AnalysisConfig(
        window_ms=analysis_settings["window_ms"],
        threshold_db=splitter["threshold_db"],
        min_gap_ms=splitter["min_gap_ms"],
        min_sample_ms=splitter["min_sample_ms"],
        montage_floor_db=analysis_settings["montage_floor_db"],
        montage_min_duration_s=analysis_settings["montage_min_duration_s"],
        montage_max_gap_count=analysis_settings["montage_max_gap_count"],
        expected_min_segments=analysis_settings["expected_min_segments"],
        expected_max_segments=analysis_settings["expected_max_segments"],
    )


def _format_track_analysis(result: analysis.TrackAnalysis) -> str:
    gap_lengths = ", ".join(f"{gap.duration_s:.2f}s" for gap in result.gaps)
    gap_summary = f"{len(result.gaps)} gaps" + (f" ({gap_lengths})" if gap_lengths else "")

    if result.track_class is analysis.TrackClass.MONTAGE:
        detail = f"montage — floor {result.noise_floor_db:.1f} dBFS, {gap_summary} — not splittable"
    else:
        detail = (
            f"splittable — floor {result.noise_floor_db:.1f} dBFS, {gap_summary}, "
            f"{len(result.segments)} samples expected"
        )

    return f"  {detail} [OUTLIER]" if result.outlier else f"  {detail}"


def _run_stub(ctx: typer.Context, input_path: Path) -> None:
    config = _load_default_config()
    typer.echo(f"{ctx.command.name}: {input_path}")
    typer.echo(config)


@app.command()
def scan(input_path: Path) -> None:
    """Print a per-file inventory and analysis report for every audio file
    in a folder: format/rate/depth, splittable-vs-montage classification,
    gap stats, and expected sample counts, plus a corpus-wide summary."""
    if not input_path.is_dir():
        typer.echo(f"Error: {input_path} is not a directory", err=True)
        raise typer.Exit(code=1)

    analysis_config = _build_analysis_config(_load_default_config())
    scanned = 0
    skipped = []
    splittable_segment_counts = []

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            info = audio_io.probe(file_path)
        except sf.LibsndfileError:
            skipped.append(file_path.name)
            continue

        try:
            result = analysis.analyze_track(audio_io.load(file_path), analysis_config)
        except sf.LibsndfileError:
            skipped.append(file_path.name)
            continue

        scanned += 1
        bit_depth = f"{info.bit_depth}-bit" if info.bit_depth is not None else "unknown bit depth"
        typer.echo(
            f"{file_path.name}: {info.format}, {info.sample_rate} Hz, "
            f"{bit_depth}, {info.channels} ch, {info.duration_s:.2f}s"
        )
        typer.echo(_format_track_analysis(result))
        if result.track_class is analysis.TrackClass.SPLITTABLE:
            splittable_segment_counts.append(len(result.segments))

    typer.echo(f"\n{scanned} file(s) scanned, {len(skipped)} skipped")
    if skipped:
        typer.echo(f"Skipped (not recognised as audio): {', '.join(skipped)}")

    if splittable_segment_counts:
        median_count = statistics.median(splittable_segment_counts)
        # Reuses the same expected-segment-count range as per-track outlier
        # flagging, so a corpus where no individual track is flagged as an
        # outlier can never contradict itself by failing this check too.
        in_range = analysis_config.expected_min_segments <= median_count <= analysis_config.expected_max_segments
        verdict = "matches" if in_range else "does not match"
        typer.echo(
            f"Corpus pattern: median {median_count:g} samples/track across splittable "
            f"tracks — {verdict} the expected ~10-samples-per-track pattern"
        )


@app.command()
def split(input_path: Path, output_path: Path) -> None:
    """Extract every detected sample from splittable tracks into individual
    FLAC files, skip montage tracks, and record every slice and skip in a
    JSON manifest written to the output directory."""
    if not input_path.is_dir():
        typer.echo(f"Error: {input_path} is not a directory", err=True)
        raise typer.Exit(code=1)
    if output_path.exists() and not output_path.is_dir():
        typer.echo(f"Error: {output_path} exists and is not a directory", err=True)
        raise typer.Exit(code=1)

    config = _load_default_config()
    analysis_config = _build_analysis_config(config)
    splitter = config["splitter"]
    head_pad_s = splitter["head_pad_ms"] / 1000
    tail_pad_s = splitter["tail_pad_ms"] / 1000

    output_path.mkdir(parents=True, exist_ok=True)
    slices: list[manifest.SliceRecord] = []
    skipped: list[manifest.SkippedRecord] = []

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            audio = audio_io.load(file_path)
        except sf.LibsndfileError:
            skipped.append(manifest.SkippedRecord(source=file_path.name, reason="unreadable"))
            typer.echo(f"{file_path.name}: skipped (unreadable)")
            continue

        result = analysis.analyze_track(audio, analysis_config)
        if result.track_class is analysis.TrackClass.MONTAGE:
            skipped.append(manifest.SkippedRecord(source=file_path.name, reason="montage"))
            typer.echo(f"{file_path.name}: montage — not splittable, skipped")
            continue

        written = 0
        for i, segment in enumerate(result.segments, start=1):
            prev_end_s = result.segments[i - 2].end_s if i > 1 else 0.0
            next_start_s = result.segments[i].start_s if i < len(result.segments) else None
            padded = analysis.pad_segment(
                segment, result.duration_s, head_pad_s, tail_pad_s, prev_end_s, next_start_s
            )
            sliced = audio_io.extract(audio, padded.start_s, padded.end_s)
            output_name = f"{file_path.name}_{i:02d}.flac"
            try:
                audio_io.write(output_path / output_name, sliced)
            except (sf.LibsndfileError, ValueError) as e:
                skipped.append(manifest.SkippedRecord(source=f"{file_path.name} (sample {i})", reason=str(e)))
                typer.echo(f"{file_path.name}: sample {i} skipped ({e})")
                continue
            slices.append(
                manifest.SliceRecord(
                    source=file_path.name,
                    start_s=padded.start_s,
                    end_s=padded.end_s,
                    output_path=output_name,
                )
            )
            written += 1
        typer.echo(f"{file_path.name}: {written} sample(s) extracted")

    manifest.write(output_path / "manifest.json", manifest.Manifest(slices=slices, skipped=skipped))
    typer.echo(f"\n{len(slices)} sample(s) written, {len(skipped)} track(s) skipped")


@app.command()
def name(ctx: typer.Context, input_path: Path) -> None:
    """Classify extracted samples and file them into the taxonomy tree."""
    _run_stub(ctx, input_path)
