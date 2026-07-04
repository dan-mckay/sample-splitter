import pytest

from sample_splitter import audio_io
from tests.fixtures import make_tone_sequence


def test_tone_sequence_duration_matches_tone_and_gap_lengths(tmp_path):
    path = tmp_path / "sequence.wav"

    make_tone_sequence(path, tone_count=3, tone_ms=200, gap_ms=500, sample_rate=44100)

    info = audio_io.probe(path)
    assert info.duration_s == pytest.approx(3 * (0.2 + 0.5))
