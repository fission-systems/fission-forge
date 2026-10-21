# Owned composition corpus and failed byte-codec experiment — 2026-10-22 KST

The expanded corpus supports a bounded from-scratch decompilation experiment;
it does **not** establish useful generated C or a DecBench score. The NLL-selected
byte model compiled **0/130** unchanged validation completions. Every failure
remains in the denominator. No candidate reached execution.

## Data

`corpus-config.json`, `corpus-manifest.json`, `corpus-lock.json` and
`corpus-audit.json` bind 96 generated expression templates and 1536 C functions.
Independent Python oracle vs compiled host C: **98,304/98,304 comparisons pass**.
All 1536 statically lifted/normalized, without claiming full x86 IR execution
parity. The existing audit removed three conflicting families and ambiguous or
duplicate rows. Released counts: **903 train / 130 validation / 188 test**;
unique C labels equal rows. Whole template families stay in one split. All ten
grammar primitives remain covered after filtering. The labels and generator are
owned Apache-2.0; external training rows are zero.

`protocol-manifest.json`, `inputs.json` and `references.json` freeze all 130
validation targets, 64 probes each, compiler/linker/runtime and evaluator hashes.
All **8320 original-source Wasm/oracle comparisons** passed before training and
again during scoring. The ten analytic input-independent guesses in
`blind-baselines.json` give 0 matches except `a | b`, which matches one target
(`expr_025_0006-O1`). These are Python uint32 behavior comparisons, not ten
executed C or model baselines. All references vary over the probe inputs.

## CUDA run

- Colab A100-SXM4 40 GB; F32 Rust/Candle 0.11.0; random initialization, seed 42.
- 5,380,608 parameters, 8192 positions, byte vocabulary 260, batch 1, LR 0.0003.
- Longest-row preflight: 6094 positions; Python seed selector verified against
  actual Rust sample ID. No truncation. Full validation every 256 updates.
- Six epochs: **5418 updates**; measured training phase **770.1 seconds**;
  observed whole-device peak **10,530 MiB**, sampled every 0.2 seconds.
- Lowest saved full-validation NLL selected before any functional scoring:
  **step 1792, NLL 1.3810069904**. Final step 5418 regressed to **1.9455952473**.
- Selected weights SHA-256:
  `932da52489de640a64b20e06e2f0d1beeb3a77240ec629096f9582385efdcea3`.
- Raw greedy generation budget 512 bytes; no prefix, source repair, wrapper or
  Markdown stripping. **0/130 compiled; 0/130 matched all probes**.
- Test model examples: **0**. No test-driven model selection. Runtime released.

`training-metrics.jsonl` retains every update; `training-report.json`,
`selection.json`, `selected-state.json`, `generation-index.json`,
`functional-report.json`, and `completions.jsonl` retain full provenance and raw
output bytes (base64). `source.json` records exact uploaded file hashes and dirty
worktree status; `executed-fresh-job.py` is extracted from that source archive.
`verification.json` records independently checked archive and selected-weight
hashes. Large model weights remain in ignored local `outputs/`, not Git.

This is a narrower grammar than the previous bootstrap corpus, with different
held-out targets. Its NLL or functional rate is not a controlled improvement
against that earlier experiment. This result motivates the separately frozen
[BPE follow-up](../../docs/composition-corpus.md); it does not justify scaling to
20B. Real-project data, full lifted-IR parity and DecBench scoring remain pending.
