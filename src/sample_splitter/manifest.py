import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SliceRecord:
    """One extracted sample: which source track it came from, its time
    offsets in that source, and where the extracted file was written."""

    source: str
    start_s: float
    end_s: float
    output_path: str


@dataclass(frozen=True)
class SkippedRecord:
    """A track that produced no output, and why."""

    source: str
    reason: str


@dataclass(frozen=True)
class Manifest:
    """The full record of one `split` run — the contract between `split`
    and later phases, and the audit trail back to source material."""

    slices: list[SliceRecord] = field(default_factory=list)
    skipped: list[SkippedRecord] = field(default_factory=list)


def write(path: Path, manifest: Manifest) -> None:
    data = {
        "slices": [asdict(s) for s in manifest.slices],
        "skipped": [asdict(s) for s in manifest.skipped],
    }
    path.write_text(json.dumps(data, indent=2))


def read(path: Path) -> Manifest:
    data = json.loads(path.read_text())
    return Manifest(
        slices=[SliceRecord(**s) for s in data["slices"]],
        skipped=[SkippedRecord(**s) for s in data["skipped"]],
    )


@dataclass(frozen=True)
class NameRecord:
    """One filed sample: which input sample it came from, the classification
    that was assigned, and where it currently lives under the `name` output
    root. `source` plus the original split/input directory is enough to
    reverse a rename — the untouched original is always still there."""

    source: str
    category: str
    subtype: str
    confidence: float
    review: bool
    output_path: str


@dataclass(frozen=True)
class NamingManifest:
    """The full record of one `name` run — every input sample's assigned
    name, category/subtype, and confidence, so a rerun with a different
    threshold can re-file without recomputing from nothing."""

    names: list[NameRecord] = field(default_factory=list)


def write_naming(path: Path, naming_manifest: NamingManifest) -> None:
    data = {"names": [asdict(n) for n in naming_manifest.names]}
    path.write_text(json.dumps(data, indent=2))


def read_naming(path: Path) -> NamingManifest:
    data = json.loads(path.read_text())
    return NamingManifest(names=[NameRecord(**n) for n in data["names"]])
