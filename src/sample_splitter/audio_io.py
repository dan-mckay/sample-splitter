from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# soundfile reports a file's encoding as a string "subtype" (e.g. "PCM_16"
# for 16-bit integer samples, "FLOAT" for 32-bit float) rather than a plain
# bit-depth number. This maps the subtypes we expect to see (from WAV/FLAC
# rips) to their bit depth. Anything not in here (e.g. compressed formats
# like MP3/Vorbis) falls back to None — handled by whoever displays it.
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
    """Metadata about an audio file, read from its header only — no sample
    data. Cheap to compute even for a large library, so this is what `scan`
    uses to build its inventory report."""

    format: str
    subtype: str
    bit_depth: int | None
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class AudioData:
    """An audio file fully decoded into memory: the actual samples plus the
    settings needed to write them back out unchanged. `samples` is a 2D
    numpy array shaped (frames, channels) — even mono files get a channels
    dimension of 1, so calling code doesn't need to special-case mono."""

    samples: np.ndarray
    sample_rate: int
    subtype: str


def probe(path: Path) -> AudioInfo:
    """Read just the header (format, rate, bit depth, duration) without
    decoding any audio. Used by `scan`, which only needs to report on files,
    not process them."""
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
    """Decode a file's full sample data into memory. Opens the file once via
    SoundFile so the sample data and its subtype come from the same read."""
    with sf.SoundFile(path) as f:
        samples = f.read(always_2d=True)
        return AudioData(samples=samples, sample_rate=f.samplerate, subtype=f.subtype)


def write(path: Path, audio: AudioData) -> None:
    """Write samples back out preserving the original sample rate and
    subtype (bit depth) — no resampling or format conversion happens here."""
    sf.write(path, audio.samples, audio.sample_rate, subtype=audio.subtype)
