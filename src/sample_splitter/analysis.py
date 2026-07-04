from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from sample_splitter.audio_io import AudioData

# RMS of true digital silence is 0, and log10(0) is undefined — this floors
# the RMS before converting to dB so a silent window reports a very quiet
# but finite level (-280 dBFS) instead of -inf or NaN.
_MIN_RMS = 1e-14


@dataclass(frozen=True)
class Gap:
    """A silent span between samples on a splittable track."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class Segment:
    """A candidate one-shot sample: the audio between two gaps."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class TrackClass(str, Enum):
    """Whether a track is a string of silence-separated one-shots
    (splittable) or a continuous demo with no useful silence (montage)."""

    SPLITTABLE = "splittable"
    MONTAGE = "montage"


@dataclass(frozen=True)
class AnalysisConfig:
    """Tunables for gap detection and track classification. Defaults mirror
    `default.toml`; the CLI builds one of these from the loaded config so
    this module stays free of TOML/file concerns."""

    window_ms: float = 10.0
    threshold_db: float = 20.0
    min_gap_ms: float = 300.0
    min_sample_ms: float = 100.0
    montage_floor_db: float = -55.0
    montage_min_duration_s: float = 30.0
    montage_max_gap_count: int = 1
    expected_min_segments: int = 5
    expected_max_segments: int = 15


@dataclass(frozen=True)
class TrackAnalysis:
    """The full analysis result for one track, as surfaced by `scan`."""

    track_class: TrackClass
    noise_floor_db: float
    duration_s: float
    gaps: list[Gap] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    outlier: bool = False


def windowed_rms_db(samples: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    """Split multi-channel samples into fixed-length time windows and return
    each window's loudness in dBFS. Channels are averaged first so multi-
    channel audio yields one loudness value per window, not one per channel.
    The final partial window (if the signal doesn't divide evenly) is measured
    over its own real samples only — zero-padding it would dilute its RMS and
    could misread trailing audio as silence."""
    mono = samples.mean(axis=1)
    window_size = max(1, round(sample_rate * window_ms / 1000))
    n_full_windows = len(mono) // window_size

    full = mono[: n_full_windows * window_size].reshape(-1, window_size)
    rms = np.sqrt(np.mean(full**2, axis=1))

    remainder = mono[n_full_windows * window_size :]
    if len(remainder):
        remainder_rms = np.sqrt(np.mean(remainder**2))
        rms = np.append(rms, remainder_rms)

    return 20 * np.log10(np.maximum(rms, _MIN_RMS))


def estimate_noise_floor_db(window_db: np.ndarray) -> float:
    """Estimate a track's noise floor from its per-window loudness values.
    Uses the 10th percentile rather than the minimum so a single unusually
    quiet window can't set the floor, and rather than the mean so the loud
    tone/music windows (the majority of the signal, not the gaps) don't drag
    it up."""
    return float(np.percentile(window_db, 10))


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Collapse a bool array into a list of (start, end, value) runs, each
    covering one contiguous stretch of windows sharing the same value."""
    runs = []
    start = 0
    current = mask[0]
    for i in range(1, len(mask)):
        if mask[i] != current:
            runs.append((start, i, bool(current)))
            start = i
            current = mask[i]
    runs.append((start, len(mask), bool(current)))
    return runs


def _merge_short_quiet_runs(quiet: np.ndarray, min_gap_windows: int) -> list[tuple[int, int, bool]]:
    """Quiet runs shorter than the minimum gap length aren't real gaps —
    reclassify them as active audio (a brief dip doesn't split a sample in
    two) and merge them into whatever surrounds them. This is also what
    makes leading/trailing gaps "just work": they're runs like any other,
    handled by the same generic pass."""
    merged: list[tuple[int, int, bool]] = []
    for start, end, is_quiet in _contiguous_runs(quiet):
        if is_quiet and (end - start) < min_gap_windows:
            is_quiet = False
        if merged and merged[-1][2] == is_quiet:
            merged[-1] = (merged[-1][0], end, is_quiet)
        else:
            merged.append((start, end, is_quiet))
    return merged


def _classify_track(duration_s: float, floor_db: float, gap_count: int, config: AnalysisConfig) -> TrackClass:
    """Montage vs splittable by majority vote across three independent
    signals (elevated floor, few gaps, long duration). Majority rather than
    requiring all three lets a track classify correctly on the two signals
    that matter most for short synthetic fixtures, where duration alone
    wouldn't be a meaningful signal."""
    montage_signals = 0
    if floor_db > config.montage_floor_db:
        montage_signals += 1
    if gap_count <= config.montage_max_gap_count:
        montage_signals += 1
    if duration_s > config.montage_min_duration_s:
        montage_signals += 1

    return TrackClass.MONTAGE if montage_signals >= 2 else TrackClass.SPLITTABLE


def _is_outlier(track_class: TrackClass, segment_count: int, gap_count: int, config: AnalysisConfig) -> bool:
    """Flag tracks that don't fit the common pattern for their class: a
    splittable track with a wildly unusual sample count, or a montage that
    unexpectedly contains real gaps."""
    if track_class == TrackClass.SPLITTABLE:
        return not (config.expected_min_segments <= segment_count <= config.expected_max_segments)
    return gap_count > 0


def analyze_track(audio: AudioData, config: AnalysisConfig = AnalysisConfig()) -> TrackAnalysis:
    """The main entry point: audio in, segments and stats out. Detects gaps
    relative to the file's own measured noise floor, then classifies the
    track and flags it as an outlier if warranted."""
    if len(audio.samples) == 0:
        # A degenerate (zero-frame) file has no loudness to measure — report
        # it rather than crash on the empty array percentile/run-length math
        # below, and flag it as an outlier since it fits no expected pattern.
        return TrackAnalysis(
            track_class=TrackClass.SPLITTABLE,
            noise_floor_db=20 * np.log10(_MIN_RMS),
            duration_s=0.0,
            outlier=True,
        )

    window_db = windowed_rms_db(audio.samples, audio.sample_rate, config.window_ms)
    floor_db = estimate_noise_floor_db(window_db)
    quiet = window_db <= floor_db + config.threshold_db

    window_s = config.window_ms / 1000
    min_gap_windows = max(1, round(config.min_gap_ms / config.window_ms))
    min_sample_s = config.min_sample_ms / 1000

    gaps: list[Gap] = []
    segments: list[Segment] = []
    for start, end, is_quiet in _merge_short_quiet_runs(quiet, min_gap_windows):
        start_s, end_s = start * window_s, end * window_s
        if is_quiet:
            gaps.append(Gap(start_s, end_s))
        elif end_s - start_s >= min_sample_s:
            segments.append(Segment(start_s, end_s))

    duration_s = len(audio.samples) / audio.sample_rate
    track_class = _classify_track(duration_s, floor_db, len(gaps), config)
    outlier = _is_outlier(track_class, len(segments), len(gaps), config)

    return TrackAnalysis(
        track_class=track_class,
        noise_floor_db=floor_db,
        duration_s=duration_s,
        gaps=gaps,
        segments=segments,
        outlier=outlier,
    )
