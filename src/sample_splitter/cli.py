import json
import shutil
import statistics
import tomllib
from importlib import resources
from pathlib import Path

import soundfile as sf
import typer

from sample_splitter import analysis, audio_io, classifier, manifest, naming

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


def _prune_empty_dirs(start: Path, root: Path) -> None:
    """Remove `start` and any now-empty ancestor directories up to (but not
    including) `root`, so demoting a sample out of `_review/` or a category
    folder doesn't leave stale empty directories behind."""
    current = start
    while current != root and current.is_dir() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def _list_audio_files(input_path: Path) -> tuple[list[tuple[Path, audio_io.AudioInfo]], list[str]]:
    """Every file directly in `input_path`, probed and split into audio
    files (with their header info, so callers that need it don't have to
    probe twice) and skipped filenames that don't decode as audio — the same
    filtering both `scan` and `name`'s standalone mode need."""
    audio_files: list[tuple[Path, audio_io.AudioInfo]] = []
    skipped: list[str] = []
    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            info = audio_io.probe(file_path)
        except sf.LibsndfileError:
            skipped.append(file_path.name)
            continue
        audio_files.append((file_path, info))
    return audio_files, skipped


def _collect_samples_to_name(input_path: Path) -> list[tuple[str, Path]]:
    """The samples to classify: a split manifest's slices when one is
    present in `input_path` (skipped tracks are excluded — they have no
    output file), otherwise every readable audio file found directly in the
    folder, so `name` also works standalone on a plain folder of one-shots."""
    split_manifest_path = input_path / "manifest.json"
    if split_manifest_path.exists():
        split_manifest = manifest.read(split_manifest_path)
        return [(s.output_path, input_path / s.output_path) for s in split_manifest.slices]

    audio_files, _skipped = _list_audio_files(input_path)
    return [(file_path.name, file_path) for file_path, _info in audio_files]


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
    splittable_segment_counts = []

    audio_files, skipped = _list_audio_files(input_path)
    for file_path, info in audio_files:
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
def name(
    input_path: Path,
    output_path: Path,
    review_threshold: float = typer.Option(None, help="Override the configured naming.review_threshold"),
) -> None:
    """Classify samples with the stub classifier and file them into
    `category/subtype/` directories under the output path, routing results
    below the confidence threshold to `_review/`. Reads a split manifest if
    `input_path` has one, otherwise scans the folder directly. Rerunning
    (optionally with a different --review-threshold) re-files samples in
    place without touching input_path, and removes the filed output for any
    sample whose source is no longer present."""
    if not input_path.is_dir():
        typer.echo(f"Error: {input_path} is not a directory", err=True)
        raise typer.Exit(code=1)
    if output_path.exists() and not output_path.is_dir():
        typer.echo(f"Error: {output_path} exists and is not a directory", err=True)
        raise typer.Exit(code=1)

    try:
        samples = _collect_samples_to_name(input_path)
    except (KeyError, json.JSONDecodeError) as e:
        typer.echo(f"Error: malformed manifest at {input_path / 'manifest.json'} ({e})", err=True)
        raise typer.Exit(code=1)

    config = _load_default_config()
    taxonomy = config["taxonomy"]
    threshold = review_threshold if review_threshold is not None else config["naming"]["review_threshold"]

    output_path.mkdir(parents=True, exist_ok=True)
    naming_manifest_path = output_path / "naming.json"
    try:
        previous = (
            manifest.read_naming(naming_manifest_path)
            if naming_manifest_path.exists()
            else manifest.NamingManifest()
        )
    except (KeyError, json.JSONDecodeError) as e:
        typer.echo(f"Error: malformed naming manifest at {naming_manifest_path} ({e})", err=True)
        raise typer.Exit(code=1)
    previous_by_source = {record.source: record for record in previous.names}

    used_indices: dict[tuple[str, str, bool], set[int]] = {}
    for record in previous.names:
        bucket = (record.category, record.subtype, record.review)
        index = naming.parse_index(record.output_path)
        if index is not None:
            used_indices.setdefault(bucket, set()).add(index)

    # A source that no longer appears (input file deleted, or dropped from
    # an updated split manifest) would otherwise leave its filed copy on
    # disk forever, untracked — and its numbering slot would still look
    # "used" to a later run, or "free" without the file actually being gone,
    # either of which risks a future unrelated sample silently overwriting
    # it. Removing the stale file and freeing its slot in the same pass
    # keeps the output tree and naming.json truthful to what's still there.
    current_sources = {source for source, _ in samples}
    removed = 0
    for vanished_source in previous_by_source.keys() - current_sources:
        stale_record = previous_by_source[vanished_source]
        stale_path = output_path / stale_record.output_path
        if stale_path.exists():
            stale_path.unlink()
            _prune_empty_dirs(stale_path.parent, output_path)
        stale_bucket = (stale_record.category, stale_record.subtype, stale_record.review)
        stale_index = naming.parse_index(stale_record.output_path)
        if stale_index is not None:
            used_indices.get(stale_bucket, set()).discard(stale_index)
        removed += 1

    backend = classifier.StubClassifier()
    new_records: list[manifest.NameRecord] = []
    filed = reviewed = 0

    for source, file_path in samples:
        try:
            audio = audio_io.load(file_path)
        except sf.LibsndfileError:
            typer.echo(f"{source}: skipped (unreadable)")
            continue
        result = backend.classify(audio, taxonomy)
        safe_category = naming.sanitize(result.category)
        safe_subtype = naming.sanitize(result.subtype)
        review = naming.is_review(result.confidence, threshold)
        bucket = (safe_category, safe_subtype, review)

        previous_record = previous_by_source.get(source)
        unchanged = previous_record is not None and (
            previous_record.category,
            previous_record.subtype,
            previous_record.review,
        ) == bucket

        if unchanged:
            output_path_str = previous_record.output_path
        else:
            bucket_indices = used_indices.setdefault(bucket, set())
            index = naming.next_index(bucket_indices)
            bucket_indices.add(index)
            output_path_str = str(naming.relative_path(safe_category, safe_subtype, index, review))

            if previous_record is not None:
                stale_path = output_path / previous_record.output_path
                if stale_path.exists():
                    stale_path.unlink()
                    _prune_empty_dirs(stale_path.parent, output_path)

            destination = output_path / output_path_str
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)

        new_records.append(
            manifest.NameRecord(
                source=source,
                category=safe_category,
                subtype=safe_subtype,
                confidence=result.confidence,
                review=review,
                output_path=output_path_str,
            )
        )
        if review:
            reviewed += 1
        else:
            filed += 1
        status = "review" if review else "filed"
        typer.echo(f"{source}: {status} -> {output_path_str} (confidence {result.confidence:.3f})")

    manifest.write_naming(naming_manifest_path, manifest.NamingManifest(names=new_records))
    typer.echo(f"\n{filed} sample(s) filed, {reviewed} sent to review, {removed} removed (source no longer present)")
