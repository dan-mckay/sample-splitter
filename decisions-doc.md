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
| **Taxonomy** | Fixed controlled vocabulary in the TOML config — category → subtypes (e.g. `drums: kick, snare, hi-hat, tom, cymbal, percussion, clap`). Classifier must pick from it or route to review. Refined against the real corpus during the #9 acceptance run (see findings below) and expected to keep evolving by editing config and re-running |
| **Output layout** | `output/<category>/<subtype>/` directories: `drums/kick/`, `drums/snare/`, `synth/stab/`, … |
| **Filenames** | Self-identifying `<subtype>_NN.flac` (`kick_01.flac`) — survives being dragged out of its folder; readable on the MPC Sample's 2.4" screen; FAT-safe (lowercase, no spaces) |
| **Ambiguous sounds** | Low-confidence results land in a flat `_review/misc_NN.flac` pool — not nested under the model's own guessed category/subtype, so a wrong guess never needs a rename to fix (found during #9: nesting review output under an unreliable guess meant fixing a miss twice). The real guess and confidence are still recorded in the manifest for every sample, so naming can be re-run with different thresholds. Never guess confidently into the clean tree |
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
- **Not tested by default:** `audio_io` (thin wrapper — covered incidentally). `classifier`'s CLAP backend has real integration tests, but they're `slow`-marked (require the model download) and excluded from the default run via `pytest`'s `addopts`; run explicitly with `pytest -m slow`.

---

## Acceptance run findings (#9)

The full pipeline (`scan` → `split` → `name --backend clap`) was run against the real "What's Next!" CD (92 tracks) and its output reviewed by ear and by eye against the taxonomy. This validated the PRD's success criteria and surfaced real, fixable gaps — summarised here rather than left scattered across PR descriptions.

**Pipeline validation**

| | |
|---|---|
| **`scan` → `split`** | All 92 tracks processed cleanly: 0 errors, 0 silent drops. 80 splittable tracks → 934 real one-shots; 12 correctly skipped as montage (clustered at the disc's end, plus a few full-length tracks elsewhere — matches the Class A/B split predicted during planning) |
| **Split accuracy** | Median 10 samples/track, matching the corpus-wide expectation. Every track flagged as a scan-mismatch outlier was individually investigated; all but one were confirmed *correct* detection of genuinely unusual content (content variety, a live-recorded instrument's higher noise floor, sounds played twice per gap) |
| **Real bug found and fixed** | One track ("Electribe 101") has ~3.9s of bit-exact digital silence at its tail, quieter than its real between-hit gaps — the 10th-percentile noise-floor estimator locked onto that trailing silence instead, collapsing 20 real drum-machine hits into 1 detected segment. Fixed with a per-file `threshold_db` override (the mechanism built in slice 5) rather than a code change; verified against the file directly |
| **Ear/hardware check** | Split sample boundaries confirmed clean (no clipped transients or truncated tails) by listening. Output loads correctly onto the Akai MPC Sample — the primary target device. Ableton verification (also named in the original success criteria) was deliberately skipped; the owner judged MPC Sample confirmation sufficient to close #9 |

**Classification tuning** — `name --backend clap`, tracked by filed/review split and median confidence across all 934 samples:

| Stage | Filed | Review | Median confidence |
|---|---|---|---|
| Baseline (bare-subtype prompts, original taxonomy) | 281 | 653 | 0.366 |
| + category-aware prompts, spelled-out abbreviations | 599 | 335 | 0.613 |
| + "clap" and "brass" (taxonomy gaps found via GM percussion map / NSynth / AudioSet) | 533 | 401 | 0.559 |
| + "bell" (found via owner's full review pass), loop/spoken simplified | 496 | 438 | 0.527 |

Filed count and median confidence don't move monotonically upward — each added category is a real, separately-validated fix (not a guess), but more valid candidate labels means more competition, which naturally softens top-1 confidence even when the *correct* label wins more often. Correctness, not raw filed rate, is the metric that matters here.

- **Category-aware prompts**: the original prompt template used only the bare subtype word (`"the sound of a stab"`), so subtypes reused across categories (`synth/stab` vs `strings/stab`) produced byte-identical prompts the model had no way to tell apart. Including category context alone raised a known-wrong sample's correct-label confidence from 0.366 to 0.559.
- **Abbreviated labels read poorly to the model**: `"ep"` (electric piano) scored 0.631 on a sample that was actually a drum break; spelling it out let the correct label win at 0.710+. Every abbreviated taxonomy label (`ep`, `hat`, `perc`, `sub`, `break`) was spelled out.
- **Taxonomy gaps found via prior art, validated against real samples**: cross-referencing the General MIDI percussion key map and the NSynth/AudioSet instrument-family taxonomies against the original list surfaced two missing categories — "clap" (validated: corrected 4 of 11 previously-mis-filed "snare" hits, 3 at 0.94+ confidence) and "brass" (validated: corrected all 3 known brass samples from a wrong label at mediocre confidence to the right one).
- **"bell" found by the owner's own listening pass**: `keys/electric_piano`, `keys/organ`, and `keys/piano` were all catching bell/chime content — one missing category, not three separate problems. Validated against 6 real review samples: 5 of 6 correctly moved to `bell/*`.
- **Taxonomy simplified per the owner's judgment, not accuracy data**: `beats`' `loop`/`drum_break`/`fill` collapsed to one `loop` subtype, and `vocal/spoken` merged into `vocal/phrase` — the owner found these indistinguishable on this library, so the extra granularity wasn't earning its keep.
- **An explicit "misc/unclassified" catch-all was tried and rejected**: hypothesis was that ambiguous one-shots being force-matched into ill-fitting instrument labels (noisy `brass`/`clap`/`strings-stab` results the owner flagged) could be siphoned off by giving the model an honest "none of these" option. Tested against 9 known-noisy samples plus 5 known-good ones: zero moved to `misc`, and the known-good ones stayed correctly filed. This is a real, informative negative result — abstract catch-all labels don't appear to have a strong enough acoustic signature for CLAP's zero-shot text matching to ever win against a concrete (even if wrong) instrument label. Confirms a genuine ceiling for prompt/taxonomy tuning on this content, not a gap still worth chasing.
- **Review output flattened**: `_review/` used to nest by the model's own (by definition, below-threshold) guessed category/subtype, meaning a wrong guess had to be fixed twice — once by judging the sample, once by renaming it out of a misleading folder. Review results now land in a flat `_review/misc_NN.flac` pool; the manifest still records the real guess for reference, but nothing about the file's location or name asserts it.

---

## Open questions

- **Taxonomy contents** — substantially exercised against the real corpus during the #9 acceptance run (see findings above), but still expected to keep evolving as more CDs are processed — editing the config and re-running `name` remains the intended workflow.

---

## Deferred to v1.1

- **Class B (montage) splitting** — onset-based chopping of continuous demos; doubtful value until the clean majority is proven. v1 skips and reports
- **Sonic descriptors in filenames** (`kick_punchy_01`) — plain `kick_01` chosen; descriptors could be added later from classifier output without re-splitting
- **Cloud/hybrid classifier backend** (Gemini — not Claude, which takes no audio input) — only if CLAP accuracy disappoints; the backend protocol is the insurance
- **Copying montage tracks into the output tree** — skip-and-report chosen over a `demos/` bucket
- **Configurable output format flag** — FLAC-only in v1
- **Exemplar-based classification** — the #9 acceptance run established that further prompt/taxonomy tuning has hit a real ceiling for some content (see findings above), and that corrections made today don't compound: fixing a sample's classification only affects that one file, so a new CD would hit the same confidence ceiling with no memory of past corrections. A real fix would store the CLAP audio embedding of samples the owner has confirmed correct, and classify new samples partly by similarity to those confirmed exemplars rather than (or blended with) pure zero-shot text-prompt matching — the only identified path for manual corrections to actually improve future runs. Not built in v1: a genuine new feature (a small stored-embeddings index, a blending strategy between text-prompt and exemplar-similarity scores), not a config change, and not yet scoped as a slice.

---

*Not applicable: Design (CLI output only), Sharing/virality, Monetisation — personal tool.*
