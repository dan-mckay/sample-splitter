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
