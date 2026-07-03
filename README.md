# sample-splitter

A Python CLI that takes audio sample files that are strung together, splits them into individual samples and categorises them by sample type using AI.

## Background

The source material is a library of audio files ripped from 1990s sample CDs. Each file bundles multiple unrelated one-shot sounds (drum hits, instrument notes, stabs) separated by gaps. The goal is to split every file into individual samples and give each one a short, descriptive, human-readable name based on what it sounds like.

## Prompt

The project brief below is the starting prompt for the planning and build work.

```markdown
# Project brief: Audio Sample Splitter & Auto-Namer

A Python CLI that takes audio files ripped from 1990s sample CDs — each file
contains multiple unrelated one-shot sounds (drum hits, instrument notes,
stabs) separated by gaps — splits them into individual sample files, and gives
each one a short descriptive name based on what it sounds like.

I'm a developer; this is partly a learning exercise. Favour a clear,
understandable structure over cleverness. Open decisions are listed at the
end — those are mine to make; don't pre-empt them.

## Known facts and known unknowns

- Hundreds of source files. I've seen WAV; there may also be FLAC. Treat the
  format mix as unverified.
- In the files I've inspected: ~10 samples per file, ~1 second of gap between
  them. I am NOT certain all files follow this layout — do not hard-code it.
- These are 90s CD rips: expect a real noise floor (hiss), so "silence"
  between samples is not digital silence. Detection thresholds must be
  relative to each file's measured noise floor, not absolute.
- Samples include fast transients (drum hits) and long decaying tails
  (cymbals, reverb). Splitting must not clip tails short.

## Architecture: one CLI, three subcommands

### `scan` — corpus reconnaissance (run this first, build it first)
Analyse the input library without writing any audio. Report per-file and
aggregate: container/codec, sample rate, bit depth, channels, duration,
estimated noise floor, detected gap count/lengths, estimated samples per
file. Output a summary that tells me whether the "10 samples, 1s gaps"
assumption holds, and flags outlier files that will need different handling.
This data drives the splitter's default tuning.

### `split` — the splitter
- Input folder → output folder of individual sample files. Never modifies or
  moves originals. Deterministic and idempotent: re-running produces the same
  result and doesn't duplicate output.
- Silence detection tunable via config/flags: threshold (dB relative to noise
  floor), minimum gap duration, minimum sample length, head/tail padding.
  Sensible defaults derived from `scan`; per-file overrides possible.
- `--dry-run` mode: report proposed split points and counts per file without
  writing audio, so I can sanity-check tuning against `scan` expectations
  (e.g. "expected ~10, detected 3" is a red flag worth surfacing).
- Preserve fidelity: same sample rate/bit depth/channels as the source; no
  resampling or normalisation unless I opt in.
- Write a manifest (JSON) mapping each output slice to its source file and
  time offsets — this is the contract between phases and my audit trail.

### `name` — the auto-namer
- Consumes the split output (via the manifest, or standalone on any folder of
  one-shots). Renames or copies each sample to a short, human-readable,
  filesystem-safe name reflecting what it sounds like (kick, snare, string
  stab, …).
- Open to AI-based classification — propose 2–3 candidate approaches (e.g.
  local audio-embedding/tagging model vs. cloud model API vs. DSP-feature
  heuristics) with cost, quality, and setup trade-offs before we pick one.
- Handle collisions (deterministic suffixes) and low-confidence results
  (an "unclassified/review" bucket rather than a confidently wrong name).
- Record the assigned name, category, and confidence in the manifest so
  renames are reviewable and reversible.

## Quality bar

- Phases run independently or chained; the manifest is the interface.
- Batch-first: minimal per-file intervention, but clear logging of what was
  skipped, flagged, or low-confidence — no silent failures or silent caps.
- Tests use small synthetic fixtures (generated tones + gaps + synthetic
  noise floor), so the test suite never depends on my real sample library.

## Decisions that are mine to make (flag them, don't make them)

1. Audio I/O and analysis libraries.
2. Silence-detection approach (energy threshold vs. anything fancier).
3. Classification approach and model for the namer.
4. Output naming scheme and folder organisation.
5. Config file format and CLI framework.
```

## Licence

MIT — see [LICENSE](LICENSE).

---

*The project brief above was generated with Claude Fable 5 (`claude-fable-5`) via Claude Code, July 2026.*
