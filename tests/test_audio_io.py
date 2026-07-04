import numpy as np
import pytest

from sample_splitter import audio_io


def test_round_trip_preserves_rate_bit_depth_and_channels_wav(tmp_path):
    samples = np.zeros((100, 2), dtype=np.float64)
    original = audio_io.AudioData(samples=samples, sample_rate=44100, subtype="PCM_16")
    path = tmp_path / "tone.wav"

    audio_io.write(path, original)
    loaded = audio_io.load(path)

    assert loaded.sample_rate == 44100
    assert loaded.subtype == "PCM_16"
    assert loaded.samples.shape == (100, 2)


def test_round_trip_preserves_rate_bit_depth_and_channels_flac(tmp_path):
    samples = np.zeros((100, 1), dtype=np.float64)
    original = audio_io.AudioData(samples=samples, sample_rate=48000, subtype="PCM_24")
    path = tmp_path / "tone.flac"

    audio_io.write(path, original)
    loaded = audio_io.load(path)

    assert loaded.sample_rate == 48000
    assert loaded.subtype == "PCM_24"
    assert loaded.samples.shape == (100, 1)


def test_probe_reports_format_bit_depth_channels_and_duration(tmp_path):
    samples = np.zeros((44100, 2), dtype=np.float64)
    audio_io.write(
        tmp_path / "tone.wav",
        audio_io.AudioData(samples=samples, sample_rate=44100, subtype="PCM_16"),
    )

    info = audio_io.probe(tmp_path / "tone.wav")

    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.bit_depth == 16
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.duration_s == pytest.approx(1.0)
