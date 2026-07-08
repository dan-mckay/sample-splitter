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
