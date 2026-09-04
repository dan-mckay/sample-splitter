import hashlib
from dataclasses import dataclass
from typing import Protocol

from sample_splitter.audio_io import AudioData


@dataclass(frozen=True)
class ClassificationResult:
    """One classifier verdict: a taxonomy category/subtype pair and a
    confidence in [0, 1) that the pair is correct."""

    category: str
    subtype: str
    confidence: float


class Classifier(Protocol):
    """The seam between `naming` and whatever does the actual listening.
    Any backend — this stub, or a future CLAP/cloud model — takes a decoded
    sample plus the taxonomy and returns one classification, so naming and
    CLI code never need to change when the backend does."""

    def classify(self, audio: AudioData, taxonomy: dict[str, list[str]]) -> ClassificationResult: ...


class StubClassifier:
    """Trivial deterministic placeholder: derives a category/subtype/
    confidence from a hash of the sample's own decoded audio, so the filing
    logic (numbering, collisions, `_review/` routing) is fully buildable and
    testable without the real model download. A real backend implements the
    same `Classifier` protocol with no changes required elsewhere."""

    def classify(self, audio: AudioData, taxonomy: dict[str, list[str]]) -> ClassificationResult:
        pairs = [(category, subtype) for category, subtypes in taxonomy.items() for subtype in subtypes]
        if not pairs:
            raise ValueError("taxonomy must not be empty")

        fingerprint = int.from_bytes(hashlib.sha256(audio.samples.tobytes()).digest()[:8], "big")
        category, subtype = pairs[fingerprint % len(pairs)]
        confidence = (fingerprint // len(pairs)) % 1000 / 1000
        return ClassificationResult(category=category, subtype=subtype, confidence=confidence)
