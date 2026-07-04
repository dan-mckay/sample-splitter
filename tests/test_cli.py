import pytest
from typer.testing import CliRunner

from sample_splitter.cli import app
from tests.fixtures import make_track, make_tone_sequence

runner = CliRunner()


def test_help_lists_all_subcommands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "split" in result.stdout
    assert "name" in result.stdout


@pytest.mark.parametrize("command", ["split", "name"])
def test_stub_prints_resolved_settings(command, tmp_path):
    result = runner.invoke(app, [command, str(tmp_path)])

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
