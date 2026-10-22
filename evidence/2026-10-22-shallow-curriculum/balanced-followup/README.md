**201/205 raw C outputs compiled; 0/205 matched all 64 functional probes.**
Lower NLL after adding training contexts did not improve functional generation.

# Added argument-role training contexts, fixed validation/test

The initial shallow experiment compiled all 205 validation outputs but matched
zero functions. Its training targets and generations all started with `a`, while
the validation targets started with `b`. This follow-up adds eight other **training
only** templates starting with `b`, preserving every existing validation/test
function. It is a second from-scratch development experiment, not a continuation
of the train-only or initial shallow checkpoint.

## Data and frozen identities

The 672 owned source functions pass **43,008** native-C/Python-oracle cases and
static lifting. The same strict audit admits **317 train / 205 validation /
112 test** rows with no quarantined cross-split family. All 112 added training
functions are new template contexts; no original holdout is admitted to training.
First-body-operand coverage is 205 `a` targets and 112 `b` targets, not a 50/50 split.

The validation and test release files are **byte-identical** to the initial
experiment. All 205 prompt, source, reference-module and oracle identities match,
as do the probes and **13,120** reference cases. `holdout-comparison.json` and
`reference-comparison.json` retain the checks. All **634** admitted functions
pass **81,152 comparisons per IR interpreter**, raw P-code and projected SSA,
on the same 64 probes under two register fills. No leakage/ambiguity threshold
is weakened. These are finite probes for the explicit pure-u32 ABI.

The new corpus lock is
`13dcab1ee6b65c880905818a28dd12b77c065c5356ea3024663a9f5760ea3eef`.
A 424-token BPE is relearned using only the new training split. Maximum sequence
is 375 positions; supervision is **12,781 BPE completion tokens per epoch**.
Architecture, parameter count, seed, LR and eight epochs remain fixed, but the
codec/data and number of updates change, so this is not a one-variable causal
ablation. The reserved test is used only for integrity/IR consistency and never
for model batches, generation, checkpoint selection or tuning.

## Model and execution plan

The Rust/Candle F32 model has **3,367,424 parameters**: four layers, hidden width
256, four heads, FF width 1024, RoPE and context 8192. Fresh seed 42, LR 0.0003,
batch 1 and **2536 updates / eight epochs** are declared before learning.
Full validation and checkpoints run every 256 updates plus final. Lowest full-
validation NLL selects the checkpoint before any functional scoring. All 205
outputs retain the existing raw greedy 512-token/512-byte cap. No C prefix,
repair, grammar constraint or label is supplied to inference.

Executed source hashes match public commit **291aacd**; execution was packaged
from dirty base e881f34. Generator/controller/codec/corpus/compiler/lifter hashes
are retained. The real corpus suite passes **14 tests**, the controller suite
reports **14 run with one optional skip**, and evaluation passes **14 tests**.
This is iterative development against the same validation set, not a final
performance estimate. The original 130-function nested task and DecBench are
not measured by this follow-up.


## Actual outcome

All **2536 updates** ran, processing **102,248 repeated supervised BPE tokens**.
The selected checkpoint is **step 1280, NLL 0.225931656**, weights hash
`e431fdf7dad04004b98471cbd1ebc1bfd57b6d9567dd04386f3b9f18f8ec7228`.
Final NLL is 0.286931248. Selection remained NLL-only before functional scoring.

All 205 inference requests completed. The raw outputs contain only **18 distinct
completions**. **201/205 compile**; four compilation failures remain in the full
denominator. The 201 executables run 12,864 cases, with **635/13,120 total cases
matching**. No function matches every case; every executable has an output
mismatch. All **13,120 original reference cases** are rechecked independently.
No output is repaired, wrapped or replaced.

The post-hoc role count changes from the initial model's 205 `a` / 0 `b` starts
to **135 `a` / 70 `b` starts**, but this does not yield a functional success.
Argument-role coverage alone is not sufficient evidence of correct variable,
operator and constant binding. It is not established as the sole cause of the
failures. Raw outputs, complete indices, hashes and per-function statuses are
retained. All exported raw hashes and both checkpoint digests were verified.

Training-command wall time is **77.05 seconds**, timed update/validation loop
**74.81 seconds**, and sampled whole-device peak **458 MiB** at 0.2-second
intervals. These samples can miss brief peaks. The longest-row preflight passes.
The L4 runtime terminates, and live checks show **238.12 CUs, usage 0/hour, zero
assignments and no sessions**, versus 238.25 before allocation. Transient status
observation timeouts reused the same VM. Account deltas are not general GPU prices.

The follow-up regresses compilability slightly (205 to 201) and leaves all-case
matches at zero. Neither model is promoted. The next investigation should
explicitly test IR-to-argument/operator/literal binding with train-only auxiliary
tasks, rather than treating another loss decrease or more repetitions as a
functional gain. New task/protocol changes need separately frozen measurements.
The reserved test, nested-expression validation and DecBench stay outside this
experiment's claims.
