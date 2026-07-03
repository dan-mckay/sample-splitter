# sample-splitter — decisions doc

A Python CLI that splits 90s sample-CD rips into individual samples and files them into a categorised, sampler-ready library · v1 MVP · macOS CLI · 2026-07-03

---

## Product

| | |
|---|---|
| **What it is** | A batch CLI (`scan` / `split` / `name`) that turns whole-track sample-CD rips into a categorised one-shot library |
| **Target user** | The author — a developer processing his own rip collection for use on an Akai MPC Sample and in Ableton; also a learning exercise, so clarity beats cleverness |
| **Problem it solves** | Each CD track bundles ~10 unrelated one-shots; manually chopping and naming hundreds of tracks is weeks of tedium |
| **Success criteria** | The full "What's Next!" CD (92 tracks) splits with correct sample counts; obvious sounds correctly categorised; output usable on the MPC Sample and in Ableton without manual renaming |
| **Platform** | Python 3.12 CLI on macOS (Apple Silicon), managed with uv |
| **Constraint** | ~34GB free on the laptop SSD — storage-lean choices throughout (small model checkpoint, FLAC output) |

---

## Corpus facts (measured, not assumed)

| | |
|---|---|
| **Library location** | `~/Music/Samples/90s-sample-cds/` — working copy on SSD; the source SD card is the untouched backup |
| **Current contents** | One CD: Zero-G "What's Next!" — 92 FLAC tracks, 16-bit/44.1kHz stereo, 278MB. More CDs to come |
| **Class A tracks** (majority) | Strings of ~10 one-shots separated by 0.5–1s gaps. Gaps are digitally silent (noise floor ≈ −90 dBFS) — the tape-hiss worry in the original brief was unfounded for this CD |
| **Class B tracks** (~10%) | Continuous demo montages (noise floor −43 to −47 dBFS, few/no gaps, 82–171s, clustered at the end of the disc). Not splittable by silence |
| **Validation** | A stdlib probe script (windowed RMS, noise-floor-relative threshold) correctly found ~10 segments in Class A tracks and identified Class B — the planned detection approach is proven against real data |

---

## Core interaction

| | |
|---|---|
| **Primary interaction** | Batch runs: `scan` (classify tracks, report gap stats) → `split` (extract samples + manifest) → `name` (classify sounds, file into taxonomy tree) |
| **First-run path** | `scan` prints per-track class, gap stats, and expected sample counts — eyeballed before splitting anything |
| **Safety model** | Sources never modified; `split` deterministic and idempotent; `--dry-run` previews split points; manifest records every slice's source + offsets and every name + confidence, so all output is traceable and reversible |
| **Session shape** | Run a phase, skim the report, adjust config, re-run |

---

## Splitter decisions

| | |
|---|---|
| **Detection** | Windowed RMS, threshold relative to per-file measured noise floor (validated during the interview) |
| **Track classification** | `scan` labels each track Class A (splittable) or Class B (montage) using duration + noise floor + gap count |
| **Class B handling** | Skip and report as "not splittable" in the manifest; no audio output. Revisit after the pipeline is proven on Class A |
| **Tunables** | Threshold (dB above floor), min gap duration, min sample length, head/tail padding — TOML config with CLI overrides; defaults derived from `scan` data |
| **Sanity check** | `split` compares detected count against `scan`'s expectation and flags mismatches |
| **Fidelity** | Preserve source sample rate / bit depth / channels; no resampling or normalisation |

---

## Namer decisions

| | |
|---|---|
| **Classifier** | Local CLAP, small checkpoint (`laion/clap-htsat-unfused`, ~600MB; ~1.5–2GB total with PyTorch) — zero-shot match against the taxonomy labels, confidence = similarity score. Free, offline, samples never leave the machine |
| **Backend seam** | Classifier lives behind a small protocol/interface so a cloud backend can be added later without rework. Note: the Claude API accepts no audio input (verified 2026-07-03) — a cloud backend means Gemini or similar |
| **Taxonomy** | Fixed controlled vocabulary in the TOML config — category → subtypes (e.g. `drums: kick, snare, hat, tom, cymbal, perc`). Classifier must pick from it or route to review. Starter tree to be drafted in the PRD; user refines by editing config and re-running |
| **Output layout** | `output/<category>/<subtype>/` directories: `drums/kick/`, `drums/snare/`, `synth/stab/`, … |
| **Filenames** | Self-identifying `<subtype>_NN.flac` (`kick_01.flac`) — survives being dragged out of its folder; readable on the MPC Sample's 2.4" screen; FAT-safe (lowercase, no spaces) |
| **Ambiguous sounds** | Low-confidence results land in `_review/` with best-guess names; confidence recorded in the manifest for every sample so naming can be re-run with different thresholds. Never guess confidently into the clean tree |
| **Output format** | FLAC 16/44.1 — half the disk of WAV (~300MB vs ~600MB per CD), lossless, read natively by both the MPC Sample and Ableton |

---

## Architecture

Six modules; the interesting logic is concentrated in two deep, pure ones (`analysis`, `naming`):

| Module | Responsibility |
|---|---|
| `audio_io` | Thin soundfile wrapper: load → samples + rate + metadata; write slices. Deliberately boring |
| `analysis` | **Deep module.** Windowed RMS, noise-floor estimation, gap detection, A/B track classification. Pure: audio in → segments + stats out |
| `manifest` | The JSON contract between phases: slices ↔ sources ↔ offsets ↔ names ↔ confidence. Dataclasses + read/write |
| `classifier` | Backend protocol + CLAP implementation: sample + taxonomy labels → (category, subtype, confidence) |
| `naming` | Classification → filesystem-safe path, numbering, collision handling, `_review/` routing. Pure logic |
| `cli` | Typer app wiring `scan`/`split`/`name`; loads TOML config (tunables + taxonomy) |

---

## Tech stack

| | |
|---|---|
| **Language / env** | Python 3.12 via uv (system Python is 3.9 — not used). `uv run sample-splitter scan …` |
| **Audio I/O** | soundfile + numpy — native WAV+FLAC read/write, exact fidelity, transparent DSP (learning goal) |
| **CLI** | Typer — subcommands from type-hinted functions |
| **Config** | TOML (stdlib `tomllib`): splitter tunables + taxonomy |
| **Classifier** | transformers + PyTorch running `laion/clap-htsat-unfused` |
| **Persistence** | JSON manifest per run |
| **Backend / hosting** | None — local CLI |

---

## Testing decisions

- Test external behaviour, not implementation: given this audio → these segments; given this classification → this path; write → read → equal.
- Fixtures are small synthetic WAVs (generated tones + gaps + synthetic noise floor) — the suite never depends on the real library.
- **Tested:** `analysis` (thorough — synthetic files with known gap positions), `naming` (collisions, numbering, FAT-unsafe characters, review routing), `manifest` (round-trip), `cli` (smoke tests via Typer's test runner).
- **Not tested by default:** `classifier` (requires the real model download; an optional slow-marked test may be added later), `audio_io` (thin wrapper — covered incidentally).

---

## Open questions

- **Taxonomy contents** — the actual category/subtype vocabulary. A starter tree goes in the PRD; expected to evolve by editing the config and re-running `name`.

---

## Deferred to v1.1

- **Class B (montage) splitting** — onset-based chopping of continuous demos; doubtful value until the clean majority is proven. v1 skips and reports
- **Sonic descriptors in filenames** (`kick_punchy_01`) — plain `kick_01` chosen; descriptors could be added later from classifier output without re-splitting
- **Cloud/hybrid classifier backend** (Gemini — not Claude, which takes no audio input) — only if CLAP accuracy disappoints; the backend protocol is the insurance
- **Copying montage tracks into the output tree** — skip-and-report chosen over a `demos/` bucket
- **Configurable output format flag** — FLAC-only in v1

---

*Not applicable: Design (CLI output only), Sharing/virality, Monetisation — personal tool.*
