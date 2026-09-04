import json

import numpy as np
import pytest
from typer.testing import CliRunner

from sample_splitter import audio_io, manifest
from sample_splitter.cli import app
from tests.fixtures import make_track, make_tone_sequence

runner = CliRunner()


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


# Tone frequencies below are pinned to specific StubClassifier outcomes
# against the real default.toml taxonomy (found by brute-force search over
# the stub's hash-derived confidence) — not arbitrary values.
_CLEAN_HZ = 220.0  # -> drums/perc, confidence 0.678 (>= 0.5 default threshold)
_CLEAN_HZ_ALT = 670.0  # -> drums/perc, confidence 0.808 (same bucket as _CLEAN_HZ)
_REVIEW_HZ = 240.0  # -> fx/noise, confidence 0.05 (< 0.5 default threshold)
_JUST_ABOVE_HZ = 340.0  # -> guitar/riff, confidence 0.588 (0.5 <= x < 0.6)


def test_name_files_a_high_confidence_sample_into_category_subtype_dir(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "drums" / "perc" / "perc_01.flac").exists()


def test_name_routes_a_low_confidence_sample_to_review(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "unsure.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "_review" / "fx" / "noise" / "noise_01.flac").exists()
    assert not (output_dir / "fx").exists()


def test_name_records_category_subtype_and_confidence_in_the_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"] == [
        {
            "source": "clean.flac",
            "category": "drums",
            "subtype": "perc",
            "confidence": 0.678,
            "review": False,
            "output_path": "drums/perc/perc_01.flac",
        }
    ]


def test_name_works_standalone_on_a_plain_folder_with_no_manifest(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "one_shot.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "drums" / "perc" / "perc_01.flac").exists()


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
    assert (output_dir / "drums" / "perc" / "perc_01.flac").exists()
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

    assert result.exit_code == 0
    assert sorted(p.name for p in (output_dir / "drums" / "perc").glob("*.flac")) == ["perc_01.flac", "perc_02.flac"]


def test_name_is_idempotent_on_rerun(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "clean.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)

    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    before = (output_dir / "drums" / "perc" / "perc_01.flac").read_bytes()
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    after = (output_dir / "drums" / "perc" / "perc_01.flac").read_bytes()

    assert before == after
    assert list((output_dir / "drums" / "perc").glob("*.flac")) == [output_dir / "drums" / "perc" / "perc_01.flac"]


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

    runner.invoke(app, ["name", str(input_dir), str(output_dir), "--review-threshold", "0.6"])
    assert (output_dir / "_review" / "guitar" / "riff" / "riff_01.flac").exists()

    runner.invoke(app, ["name", str(input_dir), str(output_dir), "--review-threshold", "0.5"])

    assert (output_dir / "guitar" / "riff" / "riff_01.flac").exists()
    assert not (output_dir / "_review" / "guitar").exists()
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"][0]["review"] is False
    assert manifest_data["names"][0]["output_path"] == "guitar/riff/riff_01.flac"


def test_name_removes_output_file_when_its_source_disappears(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "unsure.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_REVIEW_HZ)
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    assert (output_dir / "_review" / "fx" / "noise" / "noise_01.flac").exists()

    (input_dir / "unsure.flac").unlink()
    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    assert "1 removed" in result.stdout
    assert not (output_dir / "_review" / "fx").exists()
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert manifest_data["names"] == []


def test_name_reclaims_a_vanished_sources_slot_without_orphaning_or_overwriting(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    make_tone_sequence(input_dir / "a.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ)
    runner.invoke(app, ["name", str(input_dir), str(output_dir)])
    assert (output_dir / "drums" / "perc" / "perc_01.flac").exists()

    (input_dir / "a.flac").unlink()
    make_tone_sequence(input_dir / "b.flac", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=_CLEAN_HZ_ALT)
    result = runner.invoke(app, ["name", str(input_dir), str(output_dir)])

    assert result.exit_code == 0
    manifest_data = json.loads((output_dir / "naming.json").read_text())
    assert {r["source"] for r in manifest_data["names"]} == {"b.flac"}
    assert sorted(p.name for p in (output_dir / "drums" / "perc").glob("*.flac")) == ["perc_01.flac"]
    filed = audio_io.load(output_dir / "drums" / "perc" / "perc_01.flac")
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
