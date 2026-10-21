# Reversible BPE follow-up — 2026-10-22 KST

Raw generated C compiled **12/130 (9.23%)**; **0/130** matched all 64 probes.
All twelve compiled candidates emitted the **same C function** (one distinct
compilable completion). They ran on their respective targets (768 cases); only
8 of 8320 total validation
cases matched. The other 118 outputs failed compilation and remain in the full
denominator. This establishes some unmodified compilable C, **not successful
functional decompilation**, and is not a DecBench score.

The corpus is identical to the [byte trial](../2026-10-22-composition-corpus/README.md):
903 train / 130 validation / 188 test rows with held-out template families. The
BPE and byte frozen target IDs, prompt hashes, probes and reference outputs were
checked identical. Reference sources again pass 8320 Wasm/oracle comparisons.
No external training data or test model batches were used.

## Experiment

`experiment-plan.json` and `protocol-manifest.json` were frozen before model
results were scored. `tokenizer.json` was learned solely on training prompt/C
text, weighting source words eight times. Requested vocabulary cap 772, actual
vocabulary **698**; base bytes and specials retain their historical IDs.
The maximum training row shrank from 6094 to **3836** positions; train completion
tokens per epoch fell from 85,877 bytes to **56,639 BPE tokens**. No truncation.

- Colab A100-SXM4 40 GB, Rust/Candle 0.11.0, F32, random initialization seed 42.
- **5,604,864 parameters**, 8192 learned absolute positions, four decoder layers.
- Batch 1, LR 0.0003, eight epochs = **7224 updates**; full validation and saved
  checkpoints every 256 updates plus the final update.
- Longest-row preflight seed 326, `expr_044_0006-O1`, 3836 positions, sampled
  whole-device peak **4962 MiB**. Python/Rust sampler identity verified.
- Measured training phase **566.2 seconds**, sampled whole-device peak **4962 MiB**.
- Lowest saved full-validation NLL selected **before functional scoring**:
  **step 1792, NLL 1.554765649**. Final step 7224 regressed to **2.0099553115**.
  NLL is per BPE token and cannot be directly compared with the byte trial.
- Selected weights SHA-256:
  `42bfd44204eac90e511068d00a1b5d8bad6b26681d9169f1111c9802d94a6a30`.
- Raw greedy generation: at most 512 new tokens **and 512 bytes**. Whole pieces
  crossing the byte cap are withheld with `byte_limit`; no C repair or prefix.

The scorer independently reconstructs raw bytes from generated IDs and the
checkpoint's bound BPE vocabulary. `functional-report.json`, `generation-index.json`
and `completions.jsonl` retain every target, failure and raw completion. New
schema-5 checkpoints embed their codec; the CPU acceptance record verifies
resume and changed-codec rejection. Optimized encoding preserves all 903 row
statistics and the next update's full weights and Adam moments byte for byte.

The result archive, selected weights, source archive and executed driver hashes
were independently checked (`artifact-verification.json`). Executed helpers and uploaded model/trainer sources preserve the code before
the subsequent snapshot refactor and RoPE extension. Weights remain local in
ignored `outputs/`. The named runtime was stopped and no active server session
remained before the next experiment.

Tokenization, vocabulary/parameter count and epoch count changed together; this
is not a single-variable causal comparison. The improvement in compilation does
not establish semantic progress. Function-declaration and parenthesis failures
remain, motivating a separately declared position-encoding experiment. These
are development-validation results; a final test evaluation remains reserved.

Timing above is the complete training-command wall time, including corpus loading
and initial validation, from `training-resources.json`. The training report
separately records the timed update/periodic-validation loop. Memory is sampled
whole-device usage and can miss brief peaks.
