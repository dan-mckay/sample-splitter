import numpy as np
import pytest

from sample_splitter import analysis, audio_io
from tests.fixtures import make_track, make_tone_sequence


def test_windowed_rms_db_reports_known_sine_amplitude():
    # A full-scale sine has RMS = amplitude / sqrt(2), so a 1.0-amplitude
    # sine sits at 20*log10(1/sqrt(2)) ~= -3.01 dBFS. Known value lets the
    # test assert an exact number rather than just "roughly quiet/loud".
    sample_rate = 44100
    duration_s = 0.5
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    samples = np.sin(2 * np.pi * 440.0 * t)[:, None]

    db = analysis.windowed_rms_db(samples, sample_rate, window_ms=10.0)

    # Windows don't align to whole tone cycles, so per-window RMS wobbles
    # slightly around the true value with phase — allow for that wobble.
    assert db.shape == (50,)
    assert np.allclose(db, -3.01, atol=0.15)


def test_estimate_noise_floor_db_reflects_quiet_majority_not_loud_peaks():
    # Mostly-quiet windows (the gaps) with a couple of loud ones (the tones)
    # mixed in — the floor should track the quiet majority, not get dragged
    # up by the loud minority.
    window_db = np.array([-90.0] * 18 + [-3.0] * 2)

    floor = analysis.estimate_noise_floor_db(window_db)

    assert floor == pytest.approx(-90.0, abs=0.5)


def test_analyze_track_detects_gaps_and_segments_at_known_positions(tmp_path):
    path = tmp_path / "track.wav"
    make_tone_sequence(path, tone_count=2, tone_ms=200, gap_ms=500)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert len(result.segments) == 2
    assert len(result.gaps) == 2
    for segment in result.segments:
        assert segment.duration_s == pytest.approx(0.2, abs=0.02)
    for gap in result.gaps:
        assert gap.duration_s == pytest.approx(0.5, abs=0.02)


def test_analyze_track_detects_leading_and_trailing_gaps(tmp_path):
    path = tmp_path / "track.wav"
    make_track(path, [("gap", 600), ("tone", 200), ("gap", 600)])
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert len(result.segments) == 1
    assert len(result.gaps) == 2
    assert result.gaps[0].start_s == pytest.approx(0.0, abs=0.01)
    assert result.gaps[-1].end_s == pytest.approx(1.4, abs=0.01)


def test_analyze_track_ignores_gaps_shorter_than_minimum(tmp_path):
    path = tmp_path / "track.wav"
    # A 100ms gap is below the default 300ms minimum — the two tones either
    # side of it should merge into one continuous segment, not split.
    make_track(path, [("tone", 200), ("gap", 100), ("tone", 200)])
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert len(result.gaps) == 0
    assert len(result.segments) == 1
    assert result.segments[0].duration_s == pytest.approx(0.5, abs=0.02)


def test_analyze_track_drops_samples_shorter_than_minimum(tmp_path):
    path = tmp_path / "track.wav"
    # A 50ms blip is below the default 100ms minimum sample length — it
    # shouldn't be reported as a segment at all.
    make_track(path, [("gap", 600), ("tone", 50), ("gap", 600)])
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert len(result.segments) == 0
    assert len(result.gaps) == 2


def test_analyze_track_classifies_splittable_sample_string(tmp_path):
    path = tmp_path / "track.wav"
    make_tone_sequence(path, tone_count=10, tone_ms=200, gap_ms=500)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert result.track_class == analysis.TrackClass.SPLITTABLE
    assert result.outlier is False


def test_analyze_track_classifies_montage_with_elevated_floor_and_no_gaps(tmp_path):
    path = tmp_path / "track.wav"
    # Continuous "music": mostly a loud level with brief (sub-min-gap) dips
    # to a quieter background level — no dip is long enough to be a real
    # gap, but the dips are what the floor estimate (10th percentile) picks
    # up, mirroring a real montage's elevated, non-silent noise floor.
    make_track(path, [("noise", 400, -10.0), ("noise", 50, -45.0)] * 5)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert result.track_class == analysis.TrackClass.MONTAGE
    assert len(result.gaps) == 0
    assert result.noise_floor_db > -60


def test_analyze_track_flags_splittable_outlier_with_too_few_segments(tmp_path):
    path = tmp_path / "track.wav"
    make_tone_sequence(path, tone_count=1, tone_ms=200, gap_ms=500)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert result.track_class == analysis.TrackClass.SPLITTABLE
    assert result.outlier is True


def test_analyze_track_flags_montage_outlier_with_unexpected_gap(tmp_path):
    path = tmp_path / "track.wav"
    # Enough montage pattern either side that the inserted gap stays a small
    # fraction of the whole track — otherwise its true digital silence (far
    # quieter than the montage's own noise floor) would dominate the 10th
    # percentile and pull the floor estimate down with it.
    montage_pattern = [("noise", 400, -10.0), ("noise", 50, -45.0)] * 10
    make_track(path, montage_pattern + [("gap", 500)] + montage_pattern)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert result.track_class == analysis.TrackClass.MONTAGE
    assert len(result.gaps) == 1
    assert result.outlier is True


def test_analyze_track_keeps_short_splittable_track_with_near_silent_gap(tmp_path):
    path = tmp_path / "track.wav"
    # A gap that's quiet but not perfectly digitally silent (e.g. light
    # dither/hiss) shouldn't alone be enough to read as an elevated montage
    # floor — a single one-shot sample is exactly what "few gaps" always
    # looks like, so the floor threshold must have real margin below it.
    make_tone_sequence(path, tone_count=1, tone_ms=500, gap_ms=500, noise_floor_db=-55.0)
    audio = audio_io.load(path)

    result = analysis.analyze_track(audio)

    assert result.track_class == analysis.TrackClass.SPLITTABLE


def test_pad_segment_widens_start_and_end_by_head_and_tail_padding():
    segment = analysis.Segment(start_s=1.0, end_s=2.0)

    padded = analysis.pad_segment(segment, duration_s=10.0, head_pad_s=0.1, tail_pad_s=0.3)

    assert padded.start_s == pytest.approx(0.9)
    assert padded.end_s == pytest.approx(2.3)


def test_pad_segment_clamps_to_track_bounds():
    segment = analysis.Segment(start_s=0.05, end_s=9.9)

    padded = analysis.pad_segment(segment, duration_s=10.0, head_pad_s=0.5, tail_pad_s=0.5)

    assert padded.start_s == 0.0
    assert padded.end_s == 10.0


def test_pad_segment_clamps_to_neighboring_segments_to_avoid_overlap():
    # Padding (0.3s tail + 0.3s head = 0.6s) exceeds the 0.2s gap between
    # these two segments — without neighbor clamping the padded segments
    # would overlap and duplicate audio between the two extracted files.
    segment = analysis.Segment(start_s=1.0, end_s=2.0)

    padded = analysis.pad_segment(
        segment, duration_s=10.0, head_pad_s=0.3, tail_pad_s=0.3, prev_end_s=0.9, next_start_s=2.2
    )

    assert padded.start_s == 0.9
    assert padded.end_s == 2.2


def test_analyze_track_handles_zero_length_audio_without_crashing(tmp_path):
    audio = audio_io.AudioData(samples=np.zeros((0, 1)), sample_rate=44100, subtype="PCM_16")

    result = analysis.analyze_track(audio)

    assert result.duration_s == 0.0
    assert result.gaps == []
    assert result.segments == []
    assert result.outlier is True


def test_windowed_rms_db_measures_final_partial_window_without_zero_padding():
    # A loud tone running right up to the true end of the array, with a
    # sample count that isn't a whole multiple of the window size — the
    # final window shouldn't be diluted by padding zeros past the real audio.
    sample_rate = 44100
    window_ms = 10.0
    window_size = round(sample_rate * window_ms / 1000)
    t = np.arange(window_size + window_size // 2) / sample_rate
    samples = np.sin(2 * np.pi * 440.0 * t)[:, None]

    db = analysis.windowed_rms_db(samples, sample_rate, window_ms)

    assert len(db) == 2
    assert db[1] == pytest.approx(db[0], abs=0.5)
