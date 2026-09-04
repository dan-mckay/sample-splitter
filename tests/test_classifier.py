import numpy as np
import pytest

from sample_splitter import audio_io
from sample_splitter.classifier import ClapClassifier, StubClassifier
from tests.fixtures import make_tone_sequence

TAXONOMY = {"drums": ["kick", "snare"], "bass": ["sub"]}


def _load(path):
    return audio_io.load(path)


def test_classify_returns_a_category_and_subtype_from_the_taxonomy(tmp_path):
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500)
    audio = _load(tmp_path / "a.wav")

    result = StubClassifier().classify(audio, TAXONOMY)

    assert result.category in TAXONOMY
    assert result.subtype in TAXONOMY[result.category]
    assert 0.0 <= result.confidence < 1.0


def test_classify_is_deterministic_for_the_same_audio(tmp_path):
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500)
    audio = _load(tmp_path / "a.wav")

    first = StubClassifier().classify(audio, TAXONOMY)
    second = StubClassifier().classify(audio, TAXONOMY)

    assert first == second


def test_classify_varies_with_the_audio_content(tmp_path):
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=440.0)
    make_tone_sequence(tmp_path / "b.wav", tone_count=1, tone_ms=200, gap_ms=500, tone_hz=880.0)

    result_a = StubClassifier().classify(_load(tmp_path / "a.wav"), TAXONOMY)
    result_b = StubClassifier().classify(_load(tmp_path / "b.wav"), TAXONOMY)

    assert (result_a.category, result_a.subtype, result_a.confidence) != (
        result_b.category,
        result_b.subtype,
        result_b.confidence,
    )


@pytest.mark.slow
def test_clap_classify_returns_a_category_and_subtype_from_the_taxonomy(tmp_path):
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500)
    audio = _load(tmp_path / "a.wav")

    result = ClapClassifier().classify(audio, TAXONOMY)

    assert result.category in TAXONOMY
    assert result.subtype in TAXONOMY[result.category]
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.slow
def test_clap_classify_does_not_crash_on_zero_length_audio(tmp_path):
    # soundfile happily writes/reads a 0-frame file; the classifier must not
    # crash the whole `name` run on one degenerate input.
    zero_length = audio_io.AudioData(samples=np.zeros((0, 1)), sample_rate=44100, subtype="PCM_16")

    result = ClapClassifier().classify(zero_length, TAXONOMY)

    assert result.category in TAXONOMY


@pytest.mark.slow
def test_clap_classify_handles_a_sample_rate_the_model_was_not_trained_at(tmp_path):
    # laion/clap-htsat-unfused expects 48kHz; this project's fixtures and
    # real source library are 44.1kHz, so resampling must actually work.
    make_tone_sequence(tmp_path / "a.wav", tone_count=1, tone_ms=200, gap_ms=500, sample_rate=44100)
    audio = _load(tmp_path / "a.wav")

    result = ClapClassifier().classify(audio, TAXONOMY)

    assert result.category in TAXONOMY
