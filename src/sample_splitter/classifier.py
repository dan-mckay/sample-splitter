import hashlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from sample_splitter.audio_io import AudioData

_CLAP_MODEL_NAME = "laion/clap-htsat-unfused"


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


class NamingBackend(str, Enum):
    """Which `Classifier` implementation `name` uses — config-driven via
    `[naming].backend`, overridable per-run with `--backend`."""

    STUB = "stub"
    CLAP = "clap"


def _resample_mono(mono: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear-resample a mono signal to the rate CLAP's feature extractor
    expects. This is classifier input, never written out as audio, so
    audiophile-grade resampling isn't needed — plain numpy interpolation
    keeps this dependency-free."""
    if orig_sr == target_sr or len(mono) == 0:
        return mono.astype(np.float32)
    duration_s = len(mono) / orig_sr
    target_length = max(1, round(duration_s * target_sr))
    orig_times = np.linspace(0.0, duration_s, num=len(mono), endpoint=False)
    target_times = np.linspace(0.0, duration_s, num=target_length, endpoint=False)
    return np.interp(target_times, orig_times, mono).astype(np.float32)


class ClapClassifier:
    """The real backend: zero-shot scoring of a sample against the
    taxonomy's label prompts using a small local CLAP checkpoint
    (laion/clap-htsat-unfused, ~600MB; ~1.5-2GB with PyTorch). The model is
    loaded lazily on first `classify` call, not at construction, so simply
    building a `ClapClassifier` (e.g. while resolving which backend to use)
    never triggers a download. Caching/relocating the model is left entirely
    to transformers' own defaults, which already respect HF_HOME."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        print(
            f"Loading CLAP classifier ({_CLAP_MODEL_NAME}) — first run downloads "
            "~600MB and may take a few minutes; cached after that.",
            file=sys.stderr,
        )
        self._torch = torch
        self._model = ClapModel.from_pretrained(_CLAP_MODEL_NAME)
        self._model.eval()
        self._processor = ClapProcessor.from_pretrained(_CLAP_MODEL_NAME)

    def classify(self, audio: AudioData, taxonomy: dict[str, list[str]]) -> ClassificationResult:
        self._ensure_loaded()
        pairs = [(category, subtype) for category, subtypes in taxonomy.items() for subtype in subtypes]
        if not pairs:
            raise ValueError("taxonomy must not be empty")
        prompts = [f"the sound of a {subtype}" for _category, subtype in pairs]

        target_sr = self._processor.feature_extractor.sampling_rate
        mono = audio.samples.mean(axis=1).astype(np.float32)
        resampled = _resample_mono(mono, audio.sample_rate, target_sr)

        inputs = self._processor(
            text=prompts, audio=resampled, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_audio.softmax(dim=-1)[0]

        best_index = int(probs.argmax())
        category, subtype = pairs[best_index]
        return ClassificationResult(category=category, subtype=subtype, confidence=float(probs[best_index]))
