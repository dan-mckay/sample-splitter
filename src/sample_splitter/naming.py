import re
from pathlib import Path

_UNSAFE = re.compile(r"[^a-z0-9_]+")
_WHITESPACE_OR_HYPHEN = re.compile(r"[\s\-]+")
_INDEX_SUFFIX = re.compile(r"_(\d+)\.flac$")


def sanitize(text: str) -> str:
    """Make a taxonomy label or filename component FAT-safe: lowercase, no
    spaces, no characters outside a-z0-9_. Unsafe characters are replaced
    with "_" rather than dropped, so labels differing only by punctuation
    (e.g. "kick.drum" vs "kickdrum") don't collapse into the same string and
    silently share an output bucket. Falls back to "unknown" rather than an
    empty string so a fully-unsafe input never collapses a path."""
    lowered = text.strip().lower()
    with_underscores = _WHITESPACE_OR_HYPHEN.sub("_", lowered)
    cleaned = _UNSAFE.sub("_", with_underscores).strip("_")
    return cleaned or "unknown"


def next_index(used: set[int]) -> int:
    """The smallest positive index not already in `used` — deterministic
    given the same set, and never reuses a number in the set."""
    index = 1
    while index in used:
        index += 1
    return index


def relative_path(category: str, subtype: str, index: int, review: bool) -> Path:
    """The output path for one filed sample, relative to the `name` output
    root: `category/subtype/subtype_NN.flac`, or the same shape nested under
    `_review/` for low-confidence results."""
    safe_category = sanitize(category)
    safe_subtype = sanitize(subtype)
    filename = f"{safe_subtype}_{index:02d}.flac"
    parts = (safe_category, safe_subtype, filename)
    return Path("_review", *parts) if review else Path(*parts)


def is_review(confidence: float, threshold: float) -> bool:
    """Below the threshold routes to `_review/`; exactly at the threshold is
    treated as a pass, so raising the threshold to a sample's own confidence
    keeps it in the clean tree."""
    return confidence < threshold


def parse_index(output_path: str) -> int | None:
    """Recover the numeric suffix from a previously-assigned output path, so
    a rerun can seed its collision set from what's already on disk without
    re-deriving numbering from scratch."""
    match = _INDEX_SUFFIX.search(output_path)
    return int(match.group(1)) if match else None
