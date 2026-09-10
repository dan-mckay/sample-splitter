import json
import tempfile
import tomllib
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sample_splitter import audio_io, manifest, naming
from sample_splitter.classifier import StubClassifier
from sample_splitter.cli import app
from tests.fixtures import make_track, make_tone_sequence

runner = CliRunner()


def _stub_result(tone_hz):
    """What StubClassifier deterministically assigns a synthetic one-shot
    tone at this frequency, against the real packaged default taxonomy.
    Computed at test time (not hardcoded) so a taxonomy edit only means
    re-picking a frequency that still satisfies a test's confidence-band
    comment, not hand-updating scattered category/subtype path strings."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "probe.flac"
        make_tone_sequence(path, tone_count=1, tone_ms=200, gap_ms=500, tone_hz=tone_hz)
        audio = audio_io.load(path)
    with (resources.files("sample_splitter.config") / "default.toml").open("rb") as f:
        taxonomy = tomllib.load(f)["taxonomy"]
    return StubClassifier().classify(audio, taxonomy)


def _filed_path(result, index=1):
    return naming.relative_path(result.category, result.subtype, index, review=False)


def _review_path(result, index=1):
    return naming.relative_path(result.category, result.subtype, index, review=True)

_DEFAULT_SPLITTER_CONFIG = {
    "threshold_db": 20.0,
    "min_gap_ms": 300,
    "min_sample_ms": 100,
    "head_pad_ms": 10,
    "tail_pad_ms": 50,
}
_DEFAULT_ANALYSIS_CONFIG = {
    "window_ms": 10.0,
    "montage_floor_db": -55.0,
    "montage_min_duration_s": 30.0,
    "montage_max_gap_count": 1,
    "expected_min_segments": 5,
    "expected_max_segments": 15,
    "mismatch_tolerance": 0,
}


def write_config(path, splitter=None, analysis=None, file_overrides=None):
    """Write a minimal custom TOML config for `split --config`, overriding
    only the keys a test cares about — so each test states just what makes
    it different, instead of repeating every tunable."""
    splitter_values = {**_DEFAULT_SPLITTER_CONFIG, **(splitter or {})}
    analysis_values = {**_DEFAULT_ANALYSIS_CONFIG, **(analysis or {})}

    lines = ["[splitter]"]
    lines += [f"{key} = {value}" for key, value in splitter_values.items()]
    for file_name, overrides in (file_overrides or {}).items():
        lines.append(f'\n[splitter.overrides."{file_name}"]')
        lines += [f"{key} = {value}" for key, value in overrides.items()]
    lines.append("\n[analysis]")
    lines += [f"{key} = {value}" for key, value in analysis_values.items()]
    path.write_text("\n".join(lines))


def test_help_lists_all_subcommands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "split" in result.stdout
    assert "name" in result.stdout


def test_scan_reports_wav_file_inventory(tmp_path):
    make_tone_sequence(tmp_path / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "track.wav" in result.stdout
    assert "WAV" in result.stdout
    assert "44100" in result.stdout
    assert "16" in result.stdout
    assert "1 ch" in result.stdout or "1 channel" in result.stdout


def test_scan_reports_flac_file_inventory(tmp_path):
    make_tone_sequence(tmp_path / "track.flac", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "track.flac" in result.stdout
    assert "FLAC" in result.stdout


def test_scan_lists_non_audio_files_as_skipped_without_crashing(tmp_path):
    make_tone_sequence(tmp_path / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    (tmp_path / "notes.txt").write_text("not audio")

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "notes.txt" in result.stdout
    assert "1 skipped" in result.stdout


def test_scan_prints_aggregate_counts_across_multiple_files(tmp_path):
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500)
    make_tone_sequence(tmp_path / "b.flac", tone_count=1, tone_ms=200, gap_ms=500)
    (tmp_path / "readme.md").write_text("not audio")

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "2 file(s) scanned" in result.stdout
    assert "1 skipped" in result.stdout


def test_scan_reports_track_class_floor_gaps_and_expected_sample_count(tmp_path):
    make_tone_sequence(tmp_path / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "splittable" in result.stdout
    assert "dBFS" in result.stdout
    assert "2 gaps" in result.stdout
    assert "2 samples expected" in result.stdout


def test_scan_flags_montage_tracks_as_not_splittable(tmp_path):
    make_track(tmp_path / "demo.wav", [("noise", 400, -10.0), ("noise", 50, -45.0)] * 5)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "montage" in result.stdout
    assert "not splittable" in result.stdout


def test_scan_flags_outlier_tracks(tmp_path):
    make_tone_sequence(tmp_path / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "OUTLIER" in result.stdout


def test_scan_reports_corpus_matches_expected_samples_per_track_pattern(tmp_path):
    for i in range(3):
        make_tone_sequence(tmp_path / f"track_{i}.wav", tone_count=10, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "matches the expected ~10-samples-per-track pattern" in result.stdout


def test_scan_reports_corpus_does_not_match_expected_samples_per_track_pattern(tmp_path):
    for i in range(3):
        make_tone_sequence(tmp_path / f"track_{i}.wav", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "does not match the expected ~10-samples-per-track pattern" in result.stdout


def test_scan_reports_corpus_match_for_an_even_number_of_tracks(tmp_path):
    # An even splittable-track count can give statistics.median a fractional
    # result (e.g. 9.5) — the match check must handle that, not just whole
    # segment counts.
    make_tone_sequence(tmp_path / "track_0.wav", tone_count=9, tone_ms=200, gap_ms=500)
    make_tone_sequence(tmp_path / "track_1.wav", tone_count=10, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "matches the expected ~10-samples-per-track pattern" in result.stdout


def test_scan_skips_file_that_fails_to_decode_after_a_successful_probe(tmp_path):
    path = tmp_path / "corrupt.flac"
    make_tone_sequence(path, tone_count=2, tone_ms=200, gap_ms=500)
    # Truncate the body while leaving enough of the header intact that
    # soundfile's header-only probe still succeeds — the failure only shows
    # up when the sample data is actually decoded.
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 20])

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "1 skipped" in result.stdout
    assert "corrupt.flac" in result.stdout


def test_split_writes_one_flac_per_detected_sample(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert sorted(p.name for p in output_dir.glob("*.flac")) == ["track.wav_01.flac", "track.wav_02.flac"]


def test_split_does_not_collide_on_same_stem_different_extension(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    make_tone_sequence(input_dir / "track.flac", tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert sorted(p.name for p in output_dir.glob("*.flac")) == ["track.flac_01.flac", "track.wav_01.flac"]


def test_split_pads_segments_so_extracted_samples_contain_the_full_tone(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    # A leading gap so head padding has room to extend backwards without
    # clamping at the track start — otherwise the padding effect is invisible.
    make_track(input_dir / "track.wav", [("gap", 400), ("tone", 200), ("gap", 500)])
    audio = audio_io.load(input_dir / "track.wav")
    tone_start, tone_end = round(0.4 * audio.sample_rate), round(0.6 * audio.sample_rate)
    original_tone = audio.samples[tone_start:tone_end, 0]

    runner.invoke(app, ["split", str(input_dir), str(output_dir)])
    extracted = audio_io.load(output_dir / "track.wav_01.flac")

    # Defaults: 10ms head pad, 50ms tail pad — the unpadded tone is 200ms,
    # so a bare extraction would be exactly 200ms; padding must make it longer.
    assert extracted.samples.shape[0] > (tone_end - tone_start)
    head_pad_frames = round(0.01 * audio.sample_rate)
    assert np.allclose(
        extracted.samples[head_pad_frames : head_pad_frames + len(original_tone), 0],
        original_tone,
        atol=1e-4,
    )


def test_split_skips_montage_tracks_with_no_audio_output(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "demo.wav", [("noise", 400, -10.0), ("noise", 50, -45.0)] * 5)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "not splittable" in result.stdout
    assert list(output_dir.glob("*.flac")) == []
    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["skipped"] == [{"source": "demo.wav", "reason": "montage"}]
    assert manifest_data["slices"] == []


def test_split_manifest_records_every_slice_with_source_and_offsets(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert len(manifest_data["slices"]) == 2
    for record in manifest_data["slices"]:
        assert record["source"] == "track.wav"
        assert record["end_s"] > record["start_s"]
        assert (output_dir / record["output_path"]).exists()


def test_split_never_modifies_the_source_file(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    source = input_dir / "track.wav"
    make_tone_sequence(source, tone_count=2, tone_ms=200, gap_ms=500)
    before = source.read_bytes()

    runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert source.read_bytes() == before


def test_split_is_idempotent_on_rerun(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    runner.invoke(app, ["split", str(input_dir), str(output_dir)])
    first_files = {p.name: p.read_bytes() for p in output_dir.glob("*.flac")}
    runner.invoke(app, ["split", str(input_dir), str(output_dir)])
    second_files = {p.name: p.read_bytes() for p in output_dir.glob("*.flac")}

    assert second_files == first_files


def test_split_logs_unreadable_files_in_manifest_and_report(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "notes.txt").write_text("not audio")

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "notes.txt" in result.stdout
    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["skipped"] == [{"source": "notes.txt", "reason": "unreadable"}]


def test_split_reports_clean_error_when_output_path_is_an_existing_file(tmp_path):
    input_dir, output_path = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    output_path.write_text("not a directory")

    result = runner.invoke(app, ["split", str(input_dir), str(output_path)])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_split_dry_run_also_reports_clean_error_when_output_path_is_an_existing_file(tmp_path):
    # --dry-run never touches output_path, but it should still fail fast on
    # a run that's guaranteed to fail once the user drops --dry-run, rather
    # than reporting a clean preview for a run that can't actually happen.
    input_dir, output_path = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    output_path.write_text("not a directory")

    result = runner.invoke(app, ["split", str(input_dir), str(output_path), "--dry-run"])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_split_reports_clean_error_for_missing_config_file(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--config", str(tmp_path / "missing.toml")])

    assert result.exit_code == 1
    assert "could not read config file" in result.stderr


def test_split_reports_clean_error_for_config_missing_a_required_key(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    config_path = tmp_path / "incomplete.toml"
    config_path.write_text("[splitter]\nthreshold_db = 20.0\n")

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--config", str(config_path)])

    assert result.exit_code == 1
    assert "invalid config file" in result.stderr


def test_split_dry_run_summary_includes_skipped_track_count(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "demo.wav", [("noise", 400, -10.0), ("noise", 50, -45.0)] * 5)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run"])

    assert result.exit_code == 0
    assert "0 sample(s) proposed, 1 track(s) skipped, no files written (dry run)" in result.stdout


def test_scan_corpus_check_applies_mismatch_tolerance_consistently_with_outlier_flag(tmp_path, monkeypatch):
    # Same range widening `_is_outlier` applies per-track must also apply to
    # the corpus-wide median check, or the two could contradict each other:
    # a corpus where no individual track is flagged an outlier must never
    # report a corpus-level mismatch. `scan` has no --config flag of its own
    # (out of this issue's scope), so the widened tolerance is injected by
    # patching the loaded config rather than adding one just for this test.
    for i in range(3):
        make_tone_sequence(tmp_path / f"track_{i}.wav", tone_count=2, tone_ms=200, gap_ms=500)
    from sample_splitter import cli as cli_module

    original_load = cli_module._load_default_config

    def patched_load():
        config = original_load()
        config["analysis"]["mismatch_tolerance"] = 4
        return config

    monkeypatch.setattr(cli_module, "_load_default_config", patched_load)

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == 0
    assert "OUTLIER" not in result.stdout
    assert "matches the expected ~10-samples-per-track pattern" in result.stdout


def test_split_skips_a_sample_flac_cannot_encode_without_crashing_the_run(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    # FLAC can't encode a float subtype — this sample should be logged and
    # skipped rather than crashing the whole run.
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    audio = audio_io.load(input_dir / "track.wav")
    audio_io.write(input_dir / "track.wav", audio_io.AudioData(audio.samples, audio.sample_rate, subtype="FLOAT"))

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "skipped" in result.stdout
    manifest_data = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_data["skipped"][0]["source"].startswith("track.wav (sample")
    assert list(output_dir.glob("*.flac")) == []


def test_split_dry_run_writes_nothing_and_reports_proposed_counts(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run"])

    assert result.exit_code == 0
    assert "track.wav: 2 sample(s) proposed" in result.stdout
    assert not output_dir.exists()


def test_split_dry_run_does_not_touch_an_existing_output_directory(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=2, tone_ms=200, gap_ms=500)
    runner.invoke(app, ["split", str(input_dir), str(output_dir)])
    before = {p.name: p.read_bytes() for p in output_dir.iterdir()}

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run"])

    assert result.exit_code == 0
    after = {p.name: p.read_bytes() for p in output_dir.iterdir()}
    assert after == before


def test_split_dry_run_prints_proposed_split_points(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run"])

    assert result.exit_code == 0
    # Default 10ms head pad — the tone itself starts at 0s, so the proposed
    # point should start before it once padding is applied.
    assert "01: 0.000s - " in result.stdout


def test_split_min_gap_ms_flag_overrides_config_for_this_run(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    # A 200ms gap sits below the default 300ms min_gap_ms, so by default the
    # two tones merge into one segment — lowering the flag below 200ms
    # should let the gap register and split them into two.
    make_track(input_dir / "track.wav", [("tone", 200), ("gap", 200), ("tone", 200)])

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--min-gap-ms", "100"])

    assert result.exit_code == 0
    assert "track.wav: 2 sample(s) proposed" in result.stdout


def test_split_threshold_db_flag_overrides_config_for_this_run(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    # A big low-noise block anchors the measured floor around -85dBFS; the
    # -35dB "candidate gap" between the tones sits ~50dB above that floor, so
    # the default 20dB threshold reads it as active audio (one merged
    # segment). Widening the threshold to 50dB should bring it under the
    # quiet cutoff instead, splitting the run into two segments.
    make_track(
        input_dir / "track.wav",
        [("noise", 600, -80.0), ("tone", 200), ("noise", 350, -35.0), ("tone", 200)],
    )

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run"])
    assert "track.wav: 1 sample(s) proposed" in result.stdout

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--threshold-db", "50"])
    assert "track.wav: 2 sample(s) proposed" in result.stdout


def test_split_head_pad_ms_flag_widens_proposed_split_points(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "track.wav", [("gap", 600), ("tone", 200), ("gap", 600)])

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--head-pad-ms", "100"])

    assert result.exit_code == 0
    # Default head pad is 10ms (tone starts at 0.6s); a 100ms override should
    # pull the proposed start earlier than the default would.
    assert "01: 0.500s - " in result.stdout


def test_split_tail_pad_ms_flag_widens_proposed_split_points(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "track.wav", [("gap", 600), ("tone", 200), ("gap", 600)])

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--tail-pad-ms", "200"])

    assert result.exit_code == 0
    # Default tail pad is 50ms (tone ends at 0.8s); a 200ms override should
    # push the proposed end later than the default would.
    assert " - 1.000s" in result.stdout


def test_split_reads_tunable_from_custom_config_file(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "track.wav", [("tone", 200), ("gap", 200), ("tone", 200)])
    config_path = tmp_path / "custom.toml"
    write_config(config_path, splitter={"min_gap_ms": 100})

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "track.wav: 2 sample(s) proposed" in result.stdout


def test_split_cli_flag_takes_precedence_over_custom_config_value(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "track.wav", [("tone", 200), ("gap", 200), ("tone", 200)])
    config_path = tmp_path / "custom.toml"
    write_config(config_path, splitter={"min_gap_ms": 100})

    result = runner.invoke(
        app,
        [
            "split",
            str(input_dir),
            str(output_dir),
            "--dry-run",
            "--config",
            str(config_path),
            "--min-gap-ms",
            "300",
        ],
    )

    assert result.exit_code == 0
    assert "track.wav: 1 sample(s) proposed" in result.stdout


def test_split_per_file_override_applies_only_to_the_named_file(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_track(input_dir / "special.wav", [("tone", 200), ("gap", 200), ("tone", 200)])
    make_track(input_dir / "other.wav", [("tone", 200), ("gap", 200), ("tone", 200)])
    config_path = tmp_path / "custom.toml"
    write_config(config_path, file_overrides={"special.wav": {"min_gap_ms": 100}})

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--dry-run", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "special.wav: 2 sample(s) proposed" in result.stdout
    assert "other.wav: 1 sample(s) proposed" in result.stdout


def test_split_warns_when_detected_count_diverges_from_scan_expectation(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    # Default expected range is 5-15 segments; a single-tone track
    # under-splits relative to that, so it should surface a warning.
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "WARNING" in result.stdout
    assert "track.wav" in result.stdout


def test_split_mismatch_tolerance_suppresses_warning_within_slack(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav", tone_count=1, tone_ms=200, gap_ms=500)
    config_path = tmp_path / "custom.toml"
    write_config(config_path, analysis={"mismatch_tolerance": 4})

    result = runner.invoke(app, ["split", str(input_dir), str(output_dir), "--config", str(config_path)])

    assert result.exit_code == 0
    assert "WARNING" not in result.stdout


def test_split_reports_clean_error_for_missing_directory(tmp_path):
    result = runner.invoke(app, ["split", str(tmp_path / "does-not-exist"), str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_scan_reports_clean_error_for_missing_directory(tmp_path):
    result = runner.invoke(app, ["scan", str(tmp_path / "does-not-exist")])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_scan_reports_clean_error_when_given_a_file(tmp_path):
    file_path = tmp_path / "track.wav"
    make_tone_sequence(file_path, tone_count=1, tone_ms=200, gap_ms=500)

    result = runner.invoke(app, ["scan", str(file_path)])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


# Tone frequencies below are pinned to specific StubClassifier confidence
# bands against the real default.toml taxonomy (found by brute-force search)
# — not arbitrary values. A taxonomy edit can shift which bucket/confidence
# a frequency lands on; if one of these assertions starts failing, re-run
# the brute-force search (see git history for the one-liner) rather than
# hand-picking a new value. The resulting category/subtype path is resolved
# dynamically via _stub_result/_filed_path/_review_path, so taxonomy edits
# never require touching hardcoded path strings.
_CLEAN_HZ = 110.0  # confidence 0.663 (>= 0.5 default threshold)
_CLEAN_HZ_ALT = 330.0  # confidence 0.549, same bucket as _CLEAN_HZ
_REVIEW_HZ = 160.0  # confidence 0.074 (< 0.5 default threshold)
_JUST_ABOVE_HZ = 460.0  # confidence 0.543 (0.5 <= x < 0.6)
_REVIEW_HZ_DIFFERENT_BUCKET = 100.0  # confidence 0.402 (< 0.5), different guessed bucket than _REVIEW_HZ


def test_name_files_a_high_confidence_sample_into_category_subtype_dir(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / _filed_path(_stub_result(_CLEAN_HZ))).exists()


def test_name_routes_a_low_confidence_sample_to_review(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "unsure.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    review_result = _stub_result(_REVIEW_HZ)
    assert result.exit_code == 0
    assert (output_dir / _review_path(review_result)).exists()
    assert not (output_dir / review_result.category).exists()


def test_name_review_samples_share_one_flat_numbering_pool_regardless_of_guess(tmp_path):
    # Two samples the model guesses into different (low-confidence) buckets
    # still land in the same flat _review/ pool with sequential numbering —
    # the guess is recorded in the manifest, but never shapes where the
    # file ends up, so a wrong guess never needs a rename to fix.
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "a.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ)
    make_tone_sequence(
        input_dir / "b.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ_DIFFERENT_BUCKET
    )
    a_result = _stub_result(_REVIEW_HZ)
    b_result = _stub_result(_REVIEW_HZ_DIFFERENT_BUCKET)
    assert (a_result.category, a_result.subtype) != (b_result.category, b_result.subtype)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert sorted(p.name for p in (output_dir / "_review").glob("*.flac")) == ["misc_01.flac", "misc_02.flac"]
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    by_source = {r["source"]: r for r in manifest_data["names"]}
    assert by_source["a.flac"]["category"] == a_result.category
    assert by_source["b.flac"]["category"] == b_result.category


def test_name_records_category_subtype_and_confidence_in_the_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    clean_result = _stub_result(_CLEAN_HZ)
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"] == [
        {
            "source": "clean.flac",
            "category": clean_result.category,
            "subtype": clean_result.subtype,
            "confidence": clean_result.confidence,
            "review": False,
            "output_path": str(_filed_path(clean_result)),
        }
    ]


def test_name_works_standalone_on_a_plain_folder_with_no_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "one_shot.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / _filed_path(_stub_result(_CLEAN_HZ))).exists()


def test_name_reads_samples_from_a_split_manifest_when_present(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "track.wav_01.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    manifest.write(
        input_dir / "manifest.json",
        manifest.Manifest(
            slices=[
                manifest.SliceRecord(
                    source="track.wav", start_s=0.0, end_s=0.2, output_path="track.wav_01.flac"
                )
            ],
            skipped=[manifest.SkippedRecord(source="demo.wav", reason="montage")],
        ),
    )

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / _filed_path(_stub_result(_CLEAN_HZ))).exists()
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"][0]["source"] == "track.wav_01.flac"


def test_name_skips_a_manifest_entry_whose_output_file_is_missing(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    manifest.write(
        input_dir / "manifest.json",
        manifest.Manifest(
            slices=[
                manifest.SliceRecord(
                    source="track.wav", start_s=0.0, end_s=0.2, output_path="missing.flac"
                )
            ],
        ),
    )

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "skipped" in result.stdout
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"] == []


def test_name_assigns_collision_free_numbering_within_the_same_bucket(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "a.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    make_tone_sequence(input_dir / "b.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ_ALT)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    clean_result = _stub_result(_CLEAN_HZ)
    bucket_dir = output_dir / clean_result.category / clean_result.subtype
    assert result.exit_code == 0
    assert sorted(p.name for p in bucket_dir.glob("*.flac")) == [
        f"{clean_result.subtype}_01.flac",
        f"{clean_result.subtype}_02.flac",
    ]


def test_name_is_idempotent_on_rerun(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    filed_path = output_dir / _filed_path(_stub_result(_CLEAN_HZ))
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    before = filed_path.read_bytes()
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    after = filed_path.read_bytes()

    assert before == after
    assert list(filed_path.parent.glob("*.flac")) == [filed_path]


def test_name_never_modifies_the_input_sample(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    source = input_dir / "clean.flac"
    make_tone_sequence(source, tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    before = source.read_bytes()

    runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert source.read_bytes() == before


def test_name_rerunning_with_a_lower_threshold_moves_a_review_sample_into_the_clean_tree(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "borderline.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_JUST_ABOVE_HZ)
    borderline_result = _stub_result(_JUST_ABOVE_HZ)

    runner.invoke(app, ["name", str(input_dir), str(output_dir), "--review-threshold", "0.6"])
    assert (output_dir / _review_path(borderline_result)).exists()

    runner.invoke(app, ["name", str(input_dir), str(output_dir), "--review-threshold", "0.5"])

    filed_path = _filed_path(borderline_result)
    assert (output_dir / filed_path).exists()
    assert not (output_dir / _review_path(borderline_result)).exists()
    assert not (output_dir / "_review").exists()
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"][0]["review"] is False
    assert manifest_data["names"][0]["output_path"] == str(filed_path)


def test_name_removes_output_file_when_its_source_disappears(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "unsure.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ)
    review_result = _stub_result(_REVIEW_HZ)
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    assert (output_dir / _review_path(review_result)).exists()

    (input_dir / "unsure.flac").unlink()
    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "1 removed" in result.stdout
    assert not (output_dir / _review_path(review_result)).exists()
    assert not (output_dir / "_review").exists()
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"] == []


def test_name_reclaims_a_vanished_sources_slot_without_orphaning_or_overwriting(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "a.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    clean_result = _stub_result(_CLEAN_HZ)
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    assert (output_dir / _filed_path(clean_result)).exists()

    (input_dir / "a.flac").unlink()
    make_tone_sequence(input_dir / "b.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ_ALT)
    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    bucket_dir = output_dir / clean_result.category / clean_result.subtype
    assert result.exit_code == 0
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert {r["source"] for r in manifest_data["names"]} == {"b.flac"}
    assert sorted(p.name for p in bucket_dir.glob("*.flac")) == [f"{clean_result.subtype}_01.flac"]
    filed = audio_io.load(bucket_dir / f"{clean_result.subtype}_01.flac")
    original_b = audio_io.load(input_dir / "b.flac")
    assert np.allclose(filed.samples, original_b.samples, atol=1e-4)


def test_name_reports_clean_error_for_a_malformed_split_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text("not json")

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 1
    assert "malformed manifest" in result.stderr


def test_name_reports_clean_error_for_a_malformed_naming_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    output_dir.mkdir()
    (output_dir / "naming.json").write_text("{}")

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 1
    assert "malformed naming manifest" in result.stderr


def test_name_reports_clean_error_for_missing_directory(tmp_path):
    result = runner.invoke(app, ["name", str(tmp_path / "does-not-exist"), str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_name_reports_clean_error_when_output_path_is_an_existing_file(tmp_path):
    input_dir, output_path = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    output_path.write_text("not a directory")

    result = runner.invoke(app, ["name", str(input_dir), str(output_path)])

    assert result.exit_code == 1
    assert "is not a directory" in result.stderr


def test_name_defaults_to_the_stub_backend_without_touching_the_network(tmp_path):
    # No --backend flag, no CLAP-only setup — this must resolve to the
    # packaged config's naming.backend = "stub" and never attempt a model
    # load, or every other `name` test in this suite would hit the network.
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0


def test_name_backend_flag_rejects_an_unrecognised_value(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir), "--backend", "bogus"])

    assert result.exit_code != 0
