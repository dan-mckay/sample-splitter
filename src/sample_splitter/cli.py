import statistics
import tomllib
from dataclasses import replace
from importlib import resources
from pathlib import Path

import soundfile as sf
import typer

from sample_splitter import analysis, audio_io, manifest

app = typer.Typer()

_SPLITTER_TUNABLE_KEYS = ("threshold_db", "min_gap_ms", "min_sample_ms", "head_pad_ms", "tail_pad_ms")


def _parse_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_default_config() -> dict:
    default_config_path = resources.files("sample_splitter.config") / "default.toml"
    return _parse_toml(default_config_path)


def _load_config(config_path: Path | None) -> dict:
    return _load_default_config() if config_path is None else _parse_toml(config_path)


def _resolve_splitter_settings(splitter: dict, file_name: str, cli_overrides: dict) -> dict:
    """Merge one file's effective tunables from three layers, highest
    precedence first: a CLI flag (explicit for this one invocation), a
    per-file override section (explicit for this one file), then the
    config's own top-level default."""
    file_overrides = splitter.get("overrides", {}).get(file_name, {})
    settings = {}
    for key in _SPLITTER_TUNABLE_KEYS:
        if cli_overrides.get(key) is not None:
            settings[key] = cli_overrides[key]
        elif key in file_overrides:
            settings[key] = file_overrides[key]
        else:
            settings[key] = splitter[key]
    return settings


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
        mismatch_tolerance=analysis_settings["mismatch_tolerance"],
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
        low, high = analysis.expected_segment_range(analysis_config)
        in_range = low <= median_count <= high
        verdict = "matches" if in_range else "does not match"
        typer.echo(
            f"Corpus pattern: median {median_count:g} samples/track across splittable "
            f"tracks — {verdict} the expected ~10-samples-per-track pattern"
        )


@app.command()
def split(
    input_path: Path,
    output_path: Path,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report proposed split points and counts; write no audio or manifest."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Path to an alternate TOML config file (replaces the packaged default)."
    ),
    threshold_db: float | None = typer.Option(None, "--threshold-db", help="Override threshold_db for this run."),
    min_gap_ms: float | None = typer.Option(None, "--min-gap-ms", help="Override min_gap_ms for this run."),
    min_sample_ms: float | None = typer.Option(None, "--min-sample-ms", help="Override min_sample_ms for this run."),
    head_pad_ms: float | None = typer.Option(None, "--head-pad-ms", help="Override head_pad_ms for this run."),
    tail_pad_ms: float | None = typer.Option(None, "--tail-pad-ms", help="Override tail_pad_ms for this run."),
) -> None:
    """Extract every detected sample from splittable tracks into individual
    FLAC files, skip montage tracks, and record every slice and skip in a
    JSON manifest written to the output directory. Detection tunables come
    from the TOML config, overridable per-run via CLI flags or per-file via
    the config's splitter.overrides table. --dry-run reports the same
    proposed split points and counts without writing anything."""
    if not input_path.is_dir():
        typer.echo(f"Error: {input_path} is not a directory", err=True)
        raise typer.Exit(code=1)
    if output_path.exists() and not output_path.is_dir():
        typer.echo(f"Error: {output_path} exists and is not a directory", err=True)
        raise typer.Exit(code=1)

    try:
        config = _load_config(config_path)
        analysis_config = _build_analysis_config(config)
        splitter = config["splitter"]
    except OSError as e:
        typer.echo(f"Error: could not read config file {config_path}: {e.strerror or e}", err=True)
        raise typer.Exit(code=1)
    except (KeyError, tomllib.TOMLDecodeError) as e:
        typer.echo(f"Error: invalid config file {config_path}: {e}", err=True)
        raise typer.Exit(code=1)

    cli_overrides = {
        "threshold_db": threshold_db,
        "min_gap_ms": min_gap_ms,
        "min_sample_ms": min_sample_ms,
        "head_pad_ms": head_pad_ms,
        "tail_pad_ms": tail_pad_ms,
    }

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    slices: list[manifest.SliceRecord] = []
    skipped: list[manifest.SkippedRecord] = []
    proposed_total = 0

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            audio = audio_io.load(file_path)
        except sf.LibsndfileError:
            skipped.append(manifest.SkippedRecord(source=file_path.name, reason="unreadable"))
            typer.echo(f"{file_path.name}: skipped (unreadable)")
            continue

        settings = _resolve_splitter_settings(splitter, file_path.name, cli_overrides)
        file_analysis_config = replace(
            analysis_config,
            threshold_db=settings["threshold_db"],
            min_gap_ms=settings["min_gap_ms"],
            min_sample_ms=settings["min_sample_ms"],
        )
        head_pad_s = settings["head_pad_ms"] / 1000
        tail_pad_s = settings["tail_pad_ms"] / 1000

        result = analysis.analyze_track(audio, file_analysis_config)
        if result.track_class is analysis.TrackClass.MONTAGE:
            skipped.append(manifest.SkippedRecord(source=file_path.name, reason="montage"))
            typer.echo(f"{file_path.name}: montage — not splittable, skipped")
            continue

        if result.outlier:
            low, high = analysis.expected_segment_range(file_analysis_config)
            typer.echo(
                f"  WARNING: {file_path.name} — {len(result.segments)} sample(s) detected, "
                f"expected {low}-{high} (scan mismatch)"
            )

        padded_segments = []
        for i, segment in enumerate(result.segments, start=1):
            prev_end_s = result.segments[i - 2].end_s if i > 1 else 0.0
            next_start_s = result.segments[i].start_s if i < len(result.segments) else None
            padded_segments.append(
                analysis.pad_segment(segment, result.duration_s, head_pad_s, tail_pad_s, prev_end_s, next_start_s)
            )

        if dry_run:
            for i, padded in enumerate(padded_segments, start=1):
                typer.echo(f"  {i:02d}: {padded.start_s:.3f}s - {padded.end_s:.3f}s")
            typer.echo(f"{file_path.name}: {len(padded_segments)} sample(s) proposed")
            proposed_total += len(padded_segments)
            continue

        written = 0
        for i, padded in enumerate(padded_segments, start=1):
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

    if dry_run:
        typer.echo(
            f"\n{proposed_total} sample(s) proposed, {len(skipped)} track(s) skipped, no files written (dry run)"
        )
        return

    manifest.write(output_path / "manifest.json", manifest.Manifest(slices=slices, skipped=skipped))
    typer.echo(f"\n{len(slices)} sample(s) written, {len(skipped)} track(s) skipped")


@app.command()
def name(ctx: typer.Context, input_path: Path) -> None:
    """Classify extracted samples and file them into the taxonomy tree."""
    _run_stub(ctx, input_path)
