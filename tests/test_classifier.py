from sample_splitter import audio_io
from sample_splitter.classifier import StubClassifier
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
