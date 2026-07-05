from pathlib import Path

import numpy as np

from sample_splitter import audio_io


def make_tone_sequence(
    path: Path,
    tone_count: int,
    tone_ms: float,
    gap_ms: float,
    sample_rate: int = 44100,
    tone_hz: float = 440.0,
    channels: int = 1,
    noise_floor_db: float | None = None,
) -> None:
    """Write a WAV/FLAC of `tone_count` tones, each followed by a silent gap.

    Gaps are digitally silent unless `noise_floor_db` is given, in which case
    they're filled with low-amplitude noise at that level (dBFS) instead —
    mirroring the montage-track noise floors real scan data has to classify.
    """
    tone_samples = int(sample_rate * tone_ms / 1000)
    gap_samples = int(sample_rate * gap_ms / 1000)
    rng = np.random.default_rng(seed=0)

    t = np.arange(tone_samples) / sample_rate
    tone = np.sin(2 * np.pi * tone_hz * t)

    if noise_floor_db is None:
        gap = np.zeros(gap_samples)
    else:
        amplitude = 10 ** (noise_floor_db / 20)
        gap = rng.uniform(-amplitude, amplitude, size=gap_samples)

    segment = np.concatenate([tone, gap])
    mono = np.tile(segment, tone_count)
    samples = np.tile(mono[:, None], (1, channels))

    audio_io.write(path, audio_io.AudioData(samples=samples, sample_rate=sample_rate, subtype="PCM_16"))


def make_track(
    path: Path,
    parts: list[tuple[str, float] | tuple[str, float, float]],
    sample_rate: int = 44100,
    tone_hz: float = 440.0,
    channels: int = 1,
) -> None:
    """Write a WAV/FLAC built from an explicit sequence of parts, for tests
    that need edge-case layouts `make_tone_sequence`'s uniform repetition
    can't express (leading/trailing gaps, one odd-length gap or sample,
    elevated-floor montage noise).

    Each part is `(kind, duration_ms)` for "tone"/"gap", or
    `(kind, duration_ms, floor_db)` for "noise" (low-amplitude noise at the
    given dBFS level, standing in for a montage's elevated noise floor).
    """
    rng = np.random.default_rng(seed=0)
    chunks = []
    for part in parts:
        kind, duration_ms = part[0], part[1]
        n = int(sample_rate * duration_ms / 1000)
        if kind == "tone":
            t = np.arange(n) / sample_rate
            chunks.append(np.sin(2 * np.pi * tone_hz * t))
        elif kind == "gap":
            chunks.append(np.zeros(n))
        elif kind == "noise":
            amplitude = 10 ** (part[2] / 20)
            chunks.append(rng.uniform(-amplitude, amplitude, size=n))
        else:
            raise ValueError(f"unknown part kind: {kind!r}")

    mono = np.concatenate(chunks)
    samples = np.tile(mono[:, None], (1, channels))
    audio_io.write(path, audio_io.AudioData(samples=samples, sample_rate=sample_rate, subtype="PCM_16"))
