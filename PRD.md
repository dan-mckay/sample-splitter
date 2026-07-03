# PRD: Audio Sample Splitter & Auto-Namer

## Problem Statement

I own a library of audio files ripped from 1990s sample CDs (currently the Zero-G "What's Next!" compilation — 92 FLAC tracks — with more CDs to come). Each track bundles roughly ten unrelated one-shot sounds — drum hits, instrument notes, stabs — separated by short silent gaps. In this form the sounds are unusable: to play one on my Akai MPC Sample or drop it into an Ableton project, I would have to open each track in an editor, chop out every sample by hand, work out what it is by ear, name it, and file it somewhere sensible. Across hundreds of tracks that is weeks of tedium, so the library sits unused.

A complication discovered during planning: not every track is a clean string of samples. Around 10% (clustered at the end of the disc) are continuous demo montages with music beds and no silence gaps — silence-based splitting is meaningless for them, and a naive batch tool would shred them into garbage.

## Solution

A Python command-line tool with three subcommands forming a pipeline:

- **`scan`** analyses the library without writing any audio: it classifies each track as *splittable* (silence-separated one-shots) or *montage* (continuous demo), reports gap statistics, noise floors, and expected sample counts, and flags outliers. The user eyeballs this report before anything is split, and its measurements drive the splitter's default tuning.
- **`split`** extracts every detected sample from splittable tracks into individual files, driven by silence detection with a threshold relative to each file's measured noise floor. It never modifies sources, is deterministic and idempotent, supports a dry-run preview, and records every slice (source file, time offsets) in a JSON manifest. Montage tracks are skipped and reported.
- **`name`** classifies each extracted sample against a fixed, user-editable taxonomy using a local CLAP audio-classification model, then files it into a `category/subtype/` directory tree with a self-identifying filename (e.g. `drums/kick/kick_01.flac`). Low-confidence classifications go to a `_review/` folder instead of polluting the clean tree. Names, categories, and confidence scores are recorded in the manifest, so naming is reviewable, reversible, and re-runnable with different thresholds.

The output is a browsable, sampler-ready library that works on the MPC Sample (FAT-safe names, readable on its 2.4" screen, FLAC natively supported) and in Ableton's browser (searchable, descriptive filenames).

## User Stories

1. As a sample library owner, I want to scan my library and see every track classified as splittable or montage, so that I know what the splitter will process before any audio is written.
2. As a sample library owner, I want the scan report to show per-track gap counts, gap lengths, noise floor, and expected sample count, so that I can verify the tool's understanding of my files matches reality.
3. As a sample library owner, I want the scan report to show each file's format, sample rate, bit depth, channels, and duration, so that I can spot format outliers before they cause problems downstream.
4. As a sample library owner, I want scan to flag tracks that don't fit the common pattern, so that I can give unusual files special attention instead of discovering bad splits later.
5. As a sample library owner, I want the splitter's default tuning derived from scan measurements, so that I don't have to guess sensible thresholds myself.
6. As a producer, I want to split every splittable track in my library with one command, so that processing hundreds of tracks requires no per-file intervention.
7. As a producer, I want a dry-run mode that reports proposed split points and counts without writing audio, so that I can sanity-check tuning cheaply before committing.
8. As a producer, I want silence-detection sensitivity (threshold above noise floor, minimum gap, minimum sample length) tunable via a config file and command-line flags, so that I can fix bad splits by adjusting parameters rather than editing audio.
9. As a producer, I want per-file parameter overrides, so that one awkward track doesn't force me to re-tune the whole batch.
10. As a producer, I want head/tail padding around each detected sample, so that fast transients aren't clipped at the start and decaying tails (cymbals, reverb) aren't cut short.
11. As a producer, I want the splitter to warn when the detected sample count differs greatly from scan's expectation, so that mis-splits are surfaced instead of silently written.
12. As a sample library owner, I want my source files never modified or moved, so that the tool can never damage my originals.
13. As a sample library owner, I want split runs to be deterministic and idempotent, so that re-running never duplicates output or produces different results from the same input and settings.
14. As a producer, I want extracted samples to preserve the source's sample rate, bit depth, and channel count, so that no fidelity is lost in processing.
15. As a producer, I want montage tracks skipped and listed in the report and manifest, so that unsplittable material is acknowledged rather than shredded or silently dropped.
16. As a sample library owner, I want every slice recorded in a manifest with its source file and time offsets, so that any output sample can be traced back to exactly where it came from.
17. As a producer, I want each sample automatically classified and filed into a category/subtype folder tree, so that I can browse kicks, snares, and stabs directly instead of auditioning anonymous files.
18. As an MPC user, I want filenames that carry their subtype and number (e.g. `kick_01.flac`), so that a sample remains identifiable after being copied out of its folder onto a pad or into a project.
19. As an MPC user, I want filenames that are FAT-safe (lowercase, no spaces or special characters) and distinguishable in their first ~15 characters, so that they work and read correctly on the MPC Sample's microSD card and 2.4" screen.
20. As a producer, I want numbering to handle collisions deterministically, so that two similar kicks never overwrite each other and re-runs produce stable names.
21. As a producer, I want low-confidence classifications routed to a `_review/` folder with best-guess names, so that the tool never guesses confidently into my clean tree and I know exactly what needs manual sorting.
22. As a producer, I want every sample's assigned category, name, and confidence recorded in the manifest, so that I can audit the classifier's work and re-run naming with a different threshold without re-splitting.
23. As a sample library owner, I want the taxonomy defined in an editable config file, so that I can add or rename categories and re-run naming as my library grows.
24. As a laptop user with limited storage, I want classification to run locally with a small model and output written as FLAC, so that the tool fits comfortably in my remaining disk space.
25. As a privacy-conscious user, I want classification to work offline with no accounts or per-sample costs, so that my library never leaves my machine and processing more CDs stays free.
26. As a developer, I want the classifier behind a small interface, so that a cloud backend (e.g. Gemini — the Claude API accepts no audio input) can be added later without rework if local accuracy disappoints.
27. As a producer, I want to run `name` standalone on any folder of one-shots, so that the auto-namer is useful for samples that didn't come through the splitter.
28. As a producer, I want the three phases runnable independently or chained, so that I can re-run just the phase whose settings I changed.
29. As a sample library owner, I want clear logging of everything skipped, flagged, or low-confidence, so that nothing fails or gets dropped silently.
30. As a developer, I want the project managed with uv and runnable with a single command, so that setup on a fresh machine is trivial.
31. As a developer using this project to learn, I want a clear module structure with the complex logic isolated in pure, well-tested functions, so that I can understand and modify the DSP and naming logic confidently.
32. As a developer, I want the test suite to run on small synthetic audio fixtures, so that tests are fast and never depend on my personal sample library.

## Implementation Decisions

- **Pipeline**: three Typer subcommands — `scan`, `split`, `name` — sharing a TOML config (splitter tunables + taxonomy) and a JSON manifest that acts as the contract between phases and the audit trail.
- **Six modules**, with the complex logic concentrated in two deep, pure ones:
  - `audio_io` — thin soundfile wrapper: load file → samples + rate + metadata; write slices. Deliberately boring.
  - `analysis` — deep module: windowed RMS, per-file noise-floor estimation, gap detection, splittable-vs-montage track classification. Pure: audio in → segments and stats out.
  - `manifest` — dataclasses + JSON read/write for the slice/source/offset/name/confidence records.
  - `classifier` — a small backend protocol plus the CLAP implementation: sample + taxonomy labels → (category, subtype, confidence).
  - `naming` — pure logic: classification result → filesystem-safe path, numbering, collision handling, `_review/` routing.
  - `cli` — Typer app wiring the subcommands and loading config.
- **Silence detection**: windowed RMS with the threshold set relative to each file's measured noise floor (not absolute). Validated against the real corpus during planning: splittable tracks have digitally-silent gaps (≈ −90 dBFS floor); montages have elevated floors (−43 to −47 dBFS) and few gaps — duration + floor + gap count separates the classes cheaply.
- **Classifier**: local CLAP, small checkpoint (`laion/clap-htsat-unfused`, ~600MB; ~1.5–2GB total including PyTorch), zero-shot scoring against taxonomy label prompts; confidence is the similarity score, with a configurable review threshold.
- **Taxonomy**: fixed controlled vocabulary in the TOML config (category → subtypes). The classifier must choose from it or route to review. A starter tree ships with the tool (see Further Notes); users extend it by editing config and re-running `name`.
- **Output**: FLAC 16/44.1 (matching source; ~half the disk of WAV; native on both target platforms), filed as `output/<category>/<subtype>/<subtype>_<NN>.flac`, with `_review/` alongside for low-confidence samples.
- **Environment**: Python 3.12 managed by uv (system Python 3.9 is not used); dependencies are soundfile, numpy, typer, transformers + torch.
- **Safety invariants**: sources are read-only to the tool; split is deterministic and idempotent; every output is traceable through the manifest; renames are reversible via the manifest.

## Testing Decisions

- A good test exercises external behaviour, not implementation details: given this audio, these segments are detected; given this classification result, this path is produced; a manifest written then read is equal to the original. Tests must not break when internals are refactored.
- All audio fixtures are small synthetic WAVs — generated tones separated by gaps, with an optional synthetic noise floor — created by test helpers. The suite never depends on the user's real sample library.
- **Tested modules**: `analysis` (thorough — synthetic files with known gap positions and noise floors, asserting detected segments and track classification), `naming` (collisions, numbering stability, FAT-unsafe character handling, review routing), `manifest` (round-trip equality), and `cli` (smoke tests via Typer's test runner).
- **Not tested by default**: `classifier` (requires the ~600MB model download; an optional slow-marked integration test may be added later), `audio_io` (thin wrapper, covered incidentally by analysis tests).
- No prior art in the repo — this is a greenfield project; these tests establish the conventions.

## Out of Scope

- **Splitting montage/demo tracks** (onset-based chopping of continuous audio) — v1 skips and reports them; revisit once the clean majority is proven.
- **Sonic descriptors in filenames** (e.g. `kick_punchy_01`) — plain `kick_01` chosen; descriptors could be derived from classifier output later without re-splitting.
- **Cloud or hybrid classification** — only if local CLAP accuracy disappoints; the classifier protocol is the seam. (Any cloud backend means Gemini or similar — the Claude API accepts no audio input, verified 2026-07-03.)
- **Copying montage tracks into the output tree** — skip-and-report chosen over a `demos/` bucket.
- **Configurable output format** — FLAC-only in v1.
- **Normalisation, resampling, or any audio processing** beyond cutting at sample boundaries.

## Further Notes

- **Corpus ground truth** (measured during planning, not assumed): 92 FLAC tracks, 16-bit/44.1kHz stereo, 278MB, at `~/Music/Samples/90s-sample-cds/`. The "~10 samples per track with ~1s gaps" pattern holds for the splittable majority. The source SD card is the untouched backup; the tool works on the SSD copy.
- **Storage constraint**: ~34GB free on the laptop — this drove the small-checkpoint and FLAC decisions. The CLAP model cache can be relocated (e.g. to SD card via `HF_HOME`) if space becomes critical.
- **Target hardware**: Akai MPC Sample (2026; 2.4" screen, microSD storage, reads WAV/FLAC/AIFF at 16/24-bit, 44.1/48/96kHz, WAV must be plain PCM) and Ableton Live's browser.
- **Starter taxonomy** (to be shipped in the default config and refined by the user):
  - `drums`: kick, snare, hat, tom, cymbal, perc
  - `bass`: sub, synth, picked
  - `synth`: stab, pad, lead
  - `keys`: piano, organ, ep
  - `guitar`: chord, riff, note
  - `vocal`: phrase, shout, spoken
  - `fx`: riser, impact, noise, ambience
- **Build order recommendation**: `scan` first (it validates every assumption and produces the tuning data), then `split`, then `name`. Each phase is independently shippable and useful.
- Full decision history and rationale: `decisions-doc.md` in the repo root.
