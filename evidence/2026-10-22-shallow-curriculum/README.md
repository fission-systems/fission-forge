**Train-only diagnostic: 8/8 compiled and matched all 512 probes. Independent
validation: 205/205 compiled, but 0/205 matched all 64 probes.** This establishes
memorization and syntax acquisition on these tasks, not held-out decompilation
accuracy or a generally useful LLM.

# Shallow-operation curriculum — 2026-10-22 KST

Two separate from-scratch models isolate training memorization from held-out
syntax/input-role generalization. No previous failed checkpoint is continued.
[Implementation and commands](../../docs/shallow-curriculum.md) describe the
new owned grammar and bounded Colab pipeline.

## Frozen data and scope

Forty one/two-operation templates with 14 literal variants produce **560 owned
C functions**. Operators are unsigned add/subtract/multiply, AND/OR/XOR and
bounded shifts. All **35,840** host-C/Python-oracle comparisons and all 560
static lifts pass. The existing strict family/AST/behavior/machine/prompt
collision, ambiguous-label and duplicate policies are retained.

The admitted release has **205 train / 205 validation / 112 test** rows.
Validation primarily swaps argument roles: train templates start from `a`,
validation from `b`, with corresponding inner/outer operand-role swaps for two
operations. Every operator is represented in each partition before filtering.
Some compiler-collapsed sources are excluded, including the two-operation
right-shift families whose small XOR constants collapse after shifting.
There is no cross-split family quarantine, and no admission rule is relaxed.

All 522 admitted rows pass **66,816 raw-P-code/source comparisons and 66,816
projected-SSA/source comparisons**, using 64 probes and two register fill states.
This is finite-probe consistency for the explicit low-EAX pure-u32 ABI. It is
not whole-domain equivalence or support for arbitrary calls/memory. Test labels
are used only for integrity/IR consistency; test model examples remain zero.

The train-only BPE vocabulary has **424 tokens**, maximum training sequence
**375 positions** and **8021 supervised completion tokens per epoch**. The new
corpus lock is
`14d3a16bc2b0a03fdfa0c536c5896d9635423ca7530fd90a0009cc914f3d633a`.
No benchmark/external source was used; global external-evaluation overlap is
unmeasured. These easier tasks do not replace the previous 130-target nested
expression validation or establish real-project/DecBench performance.

## Train-only conditional diagnostic

Before GPU allocation, one eligible training example per operator was fixed
(`train-sanity-plan.json`). There are eight distinct reference behaviors on the
64 probes. Data bytes, selected IDs, train split, model, seed, optimizer and
fixed final step are bound. Eight original reference modules passed **512**
independent Wasm/source-oracle checks before learning.

The byte-token model has **461,056 parameters**, hidden width 128, four heads,
two layers, FF width 512, RoPE and context 2048. One batch containing all eight
examples runs **600 updates**, LR 0.003, seed 42. The saved final step is fixed
before scoring; no validation/test or functional-result selection occurs. The
smoke trainer checks reload and one-step Adam resume equivalence on copies.
Its final training loss is **0.0005775639**, from **5.5551715**, with a timed
training loop of **24.35 seconds**. Loss is not the functional score.

All eight examples are generated without a C prefix or repair, with a 128-byte/
token cap. Exact bytes and checkpoint/prompt/data identity are checked before
bounded C/Wasm execution. This result measures memorization of eight training
examples only, never held-out accuracy. The sanity checkpoint is retained
separately and does not initialize the next model.

## Independent fresh shallow model

Four layers, hidden width 256, four heads, FF width 1024, context 8192 and RoPE
use the new 424-token BPE. Seed 42, LR 0.0003, batch 1 and **1640 updates / eight
epochs** are fixed. Full validation and snapshots run every 256 updates plus
final; lowest full-validation NLL selects a saved checkpoint before scoring.
A separate longest-row preflight verifies sample identity and the 85% memory
gate. No row is truncated.

The frozen reference includes all **205 validation targets / 13,120 cases**.
Greedy decoding retains the 512-token/512-byte cap. No grammar constraint,
source repair, wrapper or label is supplied to inference. Every failed request
or C output stays in the denominator. The reserved 112 test rows never enter
model batches, generation or checkpoint selection. The validation role shift is
a deliberately narrow development task, not a representative final benchmark.

## Provenance and validation

`source-verification.json` binds the uploaded Rust/Cargo and job helpers to
public source commit **e881f34**; execution started from dirty base 1a3b270.
Executed builder, shallow generator, controllers and sanity scorer are retained.
The source corpus preserves compiler/lifter/SLA hashes. Source/result archives
and selected/final weights remain in ignored `outputs/` with digest checks.

The real corpus integrity suite passes **13 tests**. Controller/selection tests
report **14 run, one optional CPU-integration skip**; evaluation tests pass
**14 tests**, including train-only byte identity and invalid-token rejection.
The Rust model math is unchanged from the preceding tested implementation.


## Measured outcomes

| Model / evaluation scope | Examples | Raw C compiled | All-64-probe matches | Probe matches |
| --- | ---: | ---: | ---: | ---: |
| 461K, the eight training examples | 8 | 8 | **8** | **512/512** |
| 3.37M, independent argument-role validation | 205 | **205** | **0** | 868/13,120 |

The eight train-only outputs are distinct and reproduce the selected source C
without a prefix or repair. Their exact bytes and fixed step-600 checkpoint are
bound in `train-sanity-functional-report.json` and the generation index. This is
conditional training memorization, not validation accuracy.

The independent model completes all **1640 updates**, processing 64,168 repeated
BPE completion tokens. NLL-only selection chooses **step 512, NLL 0.702666771**;
final NLL is 0.805357135. Selected weights hash:
`7922bb36bc35cdc09f7630870786b50a88ce188cf855437bb1bf6e08b53d1973`.
All 205 requests return raw C successfully; only **11 distinct completions**
appear. Every candidate compiles and reaches execution, but every function has
at least one mismatching case. All **13,120 reference cases** are rechecked.
Partial case matches are retained but do not satisfy the functional gate.

The post-hoc role diagnostic finds that all 205 training targets begin their
body with `a`, all 205 validation targets with `b`, and all 205 generated outputs
with `a`. This is consistent with a role-coverage limitation, not proof that it
is the only cause (`argument-role-diagnostic.json`). The protocol intentionally
held out argument roles; these observations are development feedback, not a
representative benchmark score.

Train-only command wall time: **40.51 s**, including checkpoint/reload/optimizer
verification, sampled whole-device peak **1866 MiB**. Its timed update loop is
24.35 s. Fresh-model command wall time: **50.96 s**, update/periodic-validation
loop **48.79 s**, sampled peak **458 MiB**. Sampling interval is 0.2 s and can
miss brief peaks. The fresh longest-row preflight passes its memory/sample gate.
Actual hardware is L4; complete environment and execution-source hashes are
retained. The named runtime terminated. Live account checks after shutdown show
**238.25 CUs, usage 0/hour, zero active assignments and no sessions**, from 238.45
CUs before this run. This is an account delta, not a universal GPU price.

## Follow-up boundary

Preserve every validation/test function and add other **training-only** contexts
whose first operand is `b`, keeping the original validation templates excluded
from training. Re-run all strict leakage/ambiguity rules, require unchanged
ordered validation/test IDs, sources and IR prompts before learning, then freeze
and train a new model. This role-coverage follow-up is a separate experiment;
no existing checkpoint is promoted based on syntax alone. The original nested
130-target task, reserved tests and DecBench remain unmeasured by this run.


## Role-coverage follow-up result

The [fixed-holdout follow-up](balanced-followup/README.md) adds 112 training
contexts starting with `b`, retaining byte-identical validation/test releases.
The same 3.37M architecture trains fresh for 2536 updates; best validation NLL
falls to 0.225932. It compiles **201/205** raw outputs, matches **0/205** functions,
and matches 635/13,120 cases. Thus it does not improve the functional outcome.
All four syntax failures stay in the denominator; 201 candidates reach execution.
Generated first-body arguments include `b` on 70 cases, but variable/operator/
constant binding remains unresolved. Codec/data/update-count changes prevent
an isolated causal interpretation.

Across both runs, the established success is **8/8 conditional memorization on
training examples**, and C syntax acquisition in the easier task. Independent
functional generalization is not achieved. Source/evidence is public; large
weights stay local and no model release or DecBench score is claimed. Both L4
sessions terminate. Final live account state is **238.12 CUs, 0/hour, no active
assignments or sessions**, from 238.45 before the two-run curriculum experiment.
