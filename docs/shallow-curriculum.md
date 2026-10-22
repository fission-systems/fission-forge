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
