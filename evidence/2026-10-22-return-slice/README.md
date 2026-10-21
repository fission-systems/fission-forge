# Return-u32 SSA projection on Colab L4 — 2026-10-22 KST

**Recovered GPU generation: 130/130 requests completed, 0/130 raw C outputs
compiled, 0/130 reached functional success.** All outputs and failures are
retained. The functional gate remains unmet; lower NLL does not justify scaling.

This experiment follows the full-SSA RoPE run and frozen-checkpoint conditioning
diagnostic. It changes the input to a conservative return/control/effect slice
before training another small model from scratch. Design, limits and runnable
commands are in [the implementation document](../../docs/return-slice.md).

## Input and data admission

The new `pcode-plus-ssa-return-u32-v1` lane first builds the existing CFG/cell SSA.
It observes low EAX at return and retains dependencies of that value, every
branch, every return target and every side effect or potentially trapping
operation. Phi inputs are closed to a fixed point, including loop backedges.
Unused pure values and unobserved exit registers are removed. Dense value IDs
replace sparse IDs in the prompt; operand sizes, livein storage, CFG predecessors,
exit expressions and effect order remain. Opaque calls are refused.

Raw P-code/profile are the only projection inputs. Labels, ASTs and probe answers
are excluded. The pass is scoped to the explicit low-EAX ABI; it is not general
ABI recovery or arbitrary memory/call decompilation support. Dead trapping
arithmetic is deliberately retained and tested, as are loops and diamonds.

All 1536 pool rows projected. The existing strict corpus audit was rerun, using
transformed prompts in collision and ambiguity checks without relaxing any rule.
The frozen release preserves exact ordered **903 training / 130 validation /
188 test IDs, labels and raw lifts**. Family quarantine and release exclusions
are unchanged. `split-comparison.json` and `corpus-audit.json` retain this check.
The parent corpus lock remains unchanged; the derived lock is
`7c441e1e67bb093031bcd0bf44fc23016a9c502d0ff63115f7c1d3f23cb72726`.

All 1221 admitted rows were checked against source-oracle outputs on 64 probes
under two register fill states. Both raw P-code and projected SSA pass **156,288
comparisons per interpreter, zero failures** (`ir-semantics-report.json`).
This uses the bounded owned-corpus VM, explicit RDI/RSI/RSP/low-EAX ABI and checked
return target. Finite probes do not establish whole-domain equivalence. Test
labels are used only for integrity and IR consistency, never model batches or
checkpoint selection.

## Frozen supervision and evaluation

The new train-only lexical BPE learned 500 tokens. Maximum admitted training
sequence is **1472 byte positions / 935 BPE positions**, compared with full-SSA
6094 byte positions / 3836 BPE positions. Per-epoch completion supervision is
still **56,639 tokens**. There are no additional training functions or external
rows. The codec, statistics and all projection-row lengths/hashes are retained.

The original 130 validation sources, compiled Wasm reference modules, input
probes and all **8320 reference results** are identical to the previous RoPE
freeze (`reference-comparison.json`). Every reference is nonconstant on these
probes. The experiment plan and reference set were frozen before learning.
The original test split never enters the model or selection process.

Compilation flags, limits, raw-C assessment and Wasm worker are unchanged.
Evaluator edits only route the declared input representation and verify the
checkpoint/plan lane. This is a development-validation experiment, not DecBench
or a final-test measurement. Greedy generation receives no C prefix, reference
expression, grammar constraint or syntax repair. All failures remain in the
130-target denominator.

## Declared training protocol

F32 Rust/Candle, four layers, hidden width 256, four heads, FF width 1024, RoPE,
context 8192 and vocabulary 500 give **3,406,336 parameters**. The new vocabulary
also changes parameter count and seeded initialization, so this is not a clean
single-variable causal ablation of input slicing.

The fresh run uses seed 42, batch 1, LR 0.0003, eight epochs / **7224 updates**,
full validation and saved checkpoints every 256 updates plus final. A separate
one-update longest-row preflight selected `expr_088_0012-O1`, seed 374, length
935, and verified Python/Rust sample identity before the memory gate passed.
Lowest saved full-validation NLL chooses a checkpoint before functional scoring;
all 130 targets are generated with the existing 512-token/512-byte raw cap.

## Provenance and validation

Executed source is identified by payload hashes from a dirty worktree based on
6e49fdb; `source-verification.json` checks uploaded Rust/Cargo files against the
working source. Exact executed helper/controller and scorer/evaluator copies
are retained. Large checkpoints and source/result archives stay in ignored local
`outputs/`; final and selected weights must pass digest verification.

Rust **29 tests** and Clippy pass. Controller/integration **13 tests** pass with
the real derived corpus, including actual CPU training/resume and checkpoint
lane identity. Corpus integrity **11 tests** and evaluation **12 tests** pass.
The first functional gate remains a nonconstant held-out generated C function
matching all 64 probes, followed by repeatability and broader data before scaling.

## Software dispatch failure and complete-target recovery

The initial run completed all 7224 updates, but inference rejected **all 130**
requests with `unknown input representation`: the new lane was missing from the
generation allowlist. `initial-generation-index.json` and
`initial-functional-report.json` preserve the failures in the full denominator.
That first run did not measure the model's ability to produce C.

The corrective code adds the new lane to the allowlist without changing decoder
math, the codec, greedy settings or weights. A new CPU test trains a real
return-slice software-probe checkpoint and reaches generation, also rejecting
the wrong lane. Rust 30 tests and Clippy pass. Recovery controller tests report
13 run, one optional integration skip; evaluation tests now total 13 and include
rejection of changed step/weights/state/lane/update count/target denominator.

The retry uses the same selected step-768 weights, hash
`76cdc73bd84777607185b979e245d5bfb7f6f71baa1c9123d9ab37cc1419915c`,
with **zero optimizer updates**. All 130 prompts are retried under the original
frozen protocol. The scorer binds the original training plan, selection and
checkpoint to the separate generation-only payload and source. No target or
checkpoint is chosen using functional results. Original and recovery runs remain
separate artifacts.


## Measured outcome

| Measurement | Previous full-SSA RoPE | Return-u32 slice |
| --- | ---: | ---: |
| Parameters | 3,507,712 | 3,406,336 |
| BPE vocabulary | 698 | 500 |
| Mean / maximum train sequence | 1480 / 3836 | 504 / 935 |
| L4 training-command seconds | 616.1 | 233.85 |
| Sampled whole-device peak MiB | 4874 | 842 |
| NLL-selected step | 768 | 768 |
| Best full-validation BPE NLL | 0.531415 | 0.453237 |
| Final full-validation NLL | 0.736117 | 0.841953 |
| Compilable raw C | 16/130 | **0/130** |
| All-64-probe function matches | 0/130 | **0/130** |

Both runs use L4, the same ordered function/source sets and eight epochs; input,
codec, parameter set and initialization differ. Per-train-row completion-token
counts are equal, but this is not a causal single-variable experiment. Training
command timing includes verification/setup; the new timed update/periodic-
validation loop is 229.93 seconds. Memory is sampled every 0.2s, includes the
whole device baseline and can miss brief peaks. The new preflight passes the
85% memory gate. Shorter input improves measured resource use, while this run
regresses raw-C compilability.

The retry completed all **130 requests with exit code 0 and no timeout**.
There are **18 distinct raw completions**; the most repeated appears 17 times.
All 130 stop at **EOS**, and all 130 have unbalanced parentheses. Every candidate
produces a Clang `expected ')'` error; some also lack an expression, brace or
conditional colon. No candidate reaches execution. This failure is not caused
by the byte/token cap truncating output. `output-diagnostics.json` records each
row's raw hash, token/byte count, delimiter counts and compiler diagnostics.
`completions.jsonl` retains exact unmodified bytes as base64. No repair was tried.

All **8320 original reference cases** were independently rechecked while scoring.
The recovery used the same selected weights, verified state/model/corpus identity
and zero updates. `initial-run-artifact-verification.json` and
`recovery-artifact-verification.json` bind archives and weights; source hashes
match public training commit **518b101** and recovery commit **5d7cbb0**.

The training VM and the separate generation VM both terminated. Final live
account snapshots showed **238.45 CUs, usage 0/hour, zero active assignments and
no server sessions**, versus 238.65 CUs before this experiment. The 0.20-CU
account difference includes the software-dispatch recovery and is not a general
GPU price. Actual hardware was NVIDIA L4 23,034 MiB, CUDA 13.0, Rust 1.99.1 and
Candle 0.11.0. Transient observation failures retained each existing VM.

## Next gate

The compact input lane is implemented and probe-consistent, but this run gives
no functional generation success. Do not promote its weights or increase model
size based on NLL. Next isolate basic conditional mapping and syntax acquisition:
use a bounded training-only memorization diagnostic with multiple distinct
functions, then a shallow single-/two-operation curriculum before returning to
nested expressions. Freeze any new holdout and protocol before learning, keep
original failures and reserve the existing test split. None of that next-stage
training is claimed here. DecBench and real-project accuracy remain unmeasured.
