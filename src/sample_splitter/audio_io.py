from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_BIT_DEPTHS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


@dataclass(frozen=True)
class AudioInfo:
    format: str
    subtype: str
    bit_depth: int | None
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    subtype: str


def probe(path: Path) -> AudioInfo:
    info = sf.info(path)
    return AudioInfo(
        format=info.format,
        subtype=info.subtype,
        bit_depth=_BIT_DEPTHS.get(info.subtype),
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_s=info.duration,
    )


def load(path: Path) -> AudioData:
    with sf.SoundFile(path) as f:
        samples = f.read(always_2d=True)
        return AudioData(samples=samples, sample_rate=f.samplerate, subtype=f.subtype)


def write(path: Path, audio: AudioData) -> None:
    sf.write(path, audio.samples, audio.sample_rate, subtype=audio.subtype)
