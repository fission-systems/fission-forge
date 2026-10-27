# Shallow-operation curriculum and conditional sanity check

The return-slice trial generated 130 syntactically invalid outputs. This follow-up
first checks whether a small model can memorize **eight distinct training
functions**, then independently trains a fresh model on one/two-operation
expressions. These are separate checkpoints and tasks: the shallow model is not
initialized from the sanity model. Neither result substitutes for the old
130-function nested-expression validation, real projects or DecBench.

## Owned source grammar and strict admission

`scripts/corpus/shallow.py` declares 40 templates, each with 14 deterministic
literal variants (2..15). Operators are unsigned add/subtract/multiply, bitwise
AND/OR/XOR and bounded literal shifts. Every expression has one or two operations;
there are no conditionals, loops, calls or pointers in this corpus. The original
compiler, native/Python oracle, static x86 lifter and SSA projection are reused.

One-operation training templates observe `a`; their validation counterparts
observe `b`. Two-operation training/validation swap inner/outer argument roles.
The reserved test uses other two-operation compositions. Every operator appears
in all three splits before audit. These are narrow template/argument-role
holdouts, not evidence of broad algorithmic generalization.

All **560 functions** passed **35,840 native-C/oracle probes** and static lifting.
The existing family quarantine, canonical/erased AST, behavior, machine-code,
normalized-prompt, ambiguity and duplicate checks run unchanged. They retain
**205 train / 205 validation / 112 test** rows. Some compiler-collapsed sources
are ambiguous under the normalized input and are excluded; no policy is relaxed.
All **522** admitted rows pass **66,816 comparisons per interpreter**, raw P-code
and projected SSA, on 64 probes under two register initializations. This remains
finite-probe consistency for the explicit pure-u32 low-EAX ABI.

The train-only lexical BPE has **424 tokens**, maximum sequence **375 positions**,
and **8021 completion tokens per epoch**. No row is truncated. Smaller input and
source depth make this an easier new task; scores cannot be presented as a
controlled improvement over the previous nested-expression task.

## Eight-example train-only diagnostic

`--syntax-sanity` freezes one eligible training example per operator before GPU
allocation. The dataset bytes, train split, IDs, model, seed, learning rate and
fixed final step are hash-bound. The diagnostic model uses byte tokens, hidden
size 128, four heads, two layers, FF width 512, RoPE and context 2048. It trains
one fixed eight-example batch for **600 updates at LR 0.003**, seed 42. The smoke
trainer additionally checks checkpoint reload and a one-step optimizer-resume
comparison without changing the saved step-600 checkpoint.

All eight prompts are generated raw with a 128-token/byte budget. Their eight
observed source behaviors differ. References were independently compiled to
bounded Wasm and checked against all **512** source-oracle cases before learning.
The scorer checks checkpoint/data/prompt identity and exact output bytes, then
uses the existing bounded compiler/runtime with no C prefix, source repair or
formatting conversion. Eight successes would demonstrate training memorization
and input discrimination only. Validation and test model examples are zero.

## Independent shallow-model experiment

The fresh corpus model has four layers, hidden size 256, four heads, FF width
1024, RoPE and context 8192, with the 424-token vocabulary. It trains from random
initialization at LR 0.0003, seed 42, batch 1 for **1640 updates / eight epochs**.
Validation and checkpoints run every 256 updates plus final. Lowest full-validation
NLL chooses a saved checkpoint before functional scoring. A longest-row GPU
preflight checks sample identity and the 85% memory gate.

All 205 validation outputs use raw greedy decoding with a 512-token/512-byte cap.
The evaluator recompiles each original reference and checks its 64 probes,
retaining **13,120** cases and every generation/compile failure. The reserved
112 test examples never enter model batches, generation or checkpoint selection.

```sh
cargo build --locked --release -p fission-forge
python3 scripts/corpus/build.py --config configs/corpus/shallow-v1.json \
  --output outputs/corpus-shallow-full-new \
  --cc /opt/homebrew/opt/llvm/bin/clang --nm /opt/homebrew/opt/llvm/bin/llvm-nm
python3 scripts/corpus/project_return.py --corpus outputs/corpus-shallow-full-new \
  --output outputs/corpus-shallow-return-new --forge target/release/forge
target/release/forge corpus-semantics --corpus outputs/corpus-shallow-return-new \
  --output outputs/shallow-semantics-new
target/release/forge tokenizer-train --corpus outputs/corpus-shallow-return-new \
  --vocab-size 772 --output outputs/tokenizer-shallow-new
```

New corpus builds have their own locks and codec hashes; freeze a new protocol
using those real identities instead of overwriting recorded evidence. The
recorded run uses `configs/evaluation/shallow-functional-v1.json` and explicit
`configs/model/shallow-rope.json`:

```sh
python3 scripts/colab/run_corpus.py --corpus outputs/corpus-shallow-return-v1 \
  --syntax-sanity --fresh-epochs 8 --fresh-learning-rate 0.0003 \
  --tokenizer outputs/tokenizer-shallow-v1/tokenizer.json \
  --model-config configs/model/shallow-rope.json \
  --output outputs/colab-shallow-new --session forge-shallow-new --gpu L4

outputs/eval-venv/bin/python scripts/evaluation/train_sanity.py score \
  --results outputs/colab-shallow-new/results --frozen outputs/shallow-sanity-frozen-v1 \
  --output outputs/shallow-sanity-score-new
outputs/eval-venv/bin/python scripts/evaluation/score_gpu.py \
  --results outputs/colab-shallow-new/results --reference-evidence outputs/shallow-frozen-v1 \
  --corpus outputs/corpus-shallow-return-v1 --output outputs/shallow-score-new
```

The session has phase/global budgets and always terminates. Sanity weights are
retained separately alongside the fresh model's selected/final checkpoints.
Candle implements all model/training math; Python orchestrates data and evaluation.

## Initial outcome and fixed-holdout role-coverage follow-up

The eight-example diagnostic compiled **8/8** raw outputs and matched **512/512**
probes. The separate shallow model compiled **205/205** held-out C outputs, but
matched **0/205** functions on all probes (868/13,120 individual cases). Its
NLL-selected step was 512, NLL 0.702667. All training target bodies started with
`a`, all validation bodies with `b`, and all generated bodies with `a`. This
post-hoc observation suggests a coverage limitation; it is not a causal proof.

`configs/corpus/shallow-balanced-v2.json` keeps the original 40 templates/splits
and adds eight **training-only** two-operation templates starting with `b`.
Their operator combinations differ from every original validation/test template.
No reference function is moved into training. Strict re-audit admits **317 train /
205 validation / 112 test** rows. The term balanced in the experiment name means
role coverage is expanded, not that examples are split 50/50 between roles.

The original validation/test releases are **byte-identical**. All 205 frozen
prompt/source/reference-module/oracle identities and all 13,120 cases match the
initial freeze. The new data contains 672 source functions, all 43,008 native
source probes pass, and all 634 admitted rows pass 81,152 comparisons per IR
interpreter. No leakage/ambiguity rule is weakened.

The follow-up relearns a 424-token BPE on the new training split and uses the
same model dimensions, seed, LR and eight epochs, now **2536 updates**. The codec
IDs, dataset and update count differ, so an improvement is not a one-variable
causal ablation. A fresh model is trained; neither prior model is resumed. The
reserved test remains excluded from generation/selection. This is iterative
development on the same validation set, not a final performance estimate.

```sh
python3 scripts/colab/run_corpus.py --corpus outputs/corpus-shallow-balanced-return-v2 \
  --fresh-epochs 8 --fresh-learning-rate 0.0003 \
  --tokenizer outputs/tokenizer-shallow-balanced-v2/tokenizer.json \
  --model-config configs/model/shallow-rope.json \
  --output outputs/colab-shallow-balanced-new --session forge-shallow-balanced-new --gpu L4
```

Freeze `configs/evaluation/shallow-balanced-functional-v2.json` against the actual
new lock/codec before learning, and score every raw output with the unchanged
functional evaluator. A successful shallow case remains scoped to this tiny
grammar and finite probes; the original nested task and DecBench need separate
frozen runs.


## Recorded role-coverage outcome

The role-coverage follow-up completes all 2536 updates. NLL-only selection chooses
step 1280 at 0.225932; final NLL is 0.286931. The same 205 validation targets give
**201/205 raw compilations and 0/205 all-case matches**, with 635/13,120 individual
matches. Four syntax failures remain in the denominator. This does not improve
functional generation despite lower NLL.

Generated body starts shift from 205 `a` / 0 `b` to 135 `a` / 70 `b`, but correct
argument/operator/literal binding is still not established. The positive result
from this curriculum remains **8/8 training-example memorization**, separately
measured, plus basic C syntax acquisition. Neither independent model is promoted.
[Complete follow-up evidence](../evidence/2026-10-22-shallow-curriculum/balanced-followup/README.md)
retains all raw outputs, failures, source/archive/weight hashes and the unchanged
holdout proof. Next isolate binding through explicit training-only auxiliary
tasks before further repetitions or model-size increases.

## Wide-literal follow-up (not supported)

`configs/corpus/shallow-wide-v3.json` adds 1,090 training-only variants with
distinct constants outside 2..15 while keeping validation/test bytes identical
(1367 / 205 / 112 rows). A fresh 3.37M model trained for 10,936 updates compiled
**188/205** and matched **1/205** functions (904/13,120 cases); constants agree
with the reference in only 15/205 outputs and 9 of 15 validation families still
emit one output for every constant. The hypothesis that constant diversity alone
fixes input copying is not supported, and the one match is not promoted. See
[evidence](../evidence/2026-10-26-wide-constant/README.md).

A same-corpus re-run at LR 0.003 compiled 86/205 and matched 0/205 (constants 7/205), so a higher LR does not recover input copying; see the same evidence directory.

## Compact-lane follow-up (partial improvement)

`pcode-ssa-return-u32-compact-v1` drops the constant ABI plumbing of the
return-u32 prompt for straight-line pure functions (810 to 497 max tokens) while
keeping the same 1367 / 205 / 112 rows. With the same model and updates it compiles
**203/205** (was 188), reads the operator in **174/205** (was 87) and matches
3/205 functions, but those three coincide with a fixed literal 7 the model emits
in 156/205 outputs. Literal copying and argument-role binding remain unsolved.
See [evidence](../evidence/2026-10-28-compact-lane/README.md).

## Pretrained-model comparison

Fine-tuning `Qwen/Qwen2.5-Coder-0.5B` (Apache-2.0) for 3 epochs on the same
compact-lane rows compiles **205/205** validation outputs and matches **79/205**
functions with the NLL-selected checkpoint (97/205 at the final step; scratch
model: 3/205, coincidental). It copies literals and composes held-out argument
roles for two-operation functions (76-97 of 98) but wraps one-operation functions in
the training prior `<< 3u`. See
[evidence](../evidence/2026-10-28-pretrained-finetune/README.md).
