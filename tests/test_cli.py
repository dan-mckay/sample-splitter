import json

import numpy as np
import pytest
from typer.testing import CliRunner

from sample_splitter import audio_io
from sample_splitter.cli import app
from tests.fixtures import make_track, make_tone_sequence

runner = CliRunner()

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


def test_stub_prints_resolved_settings(tmp_path):
    result = runner.invoke(app, ["name", str(tmp_path)])

    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout
    assert "threshold_db" in result.stdout
    assert "taxonomy" in result.stdout


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
