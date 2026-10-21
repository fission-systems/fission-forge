# Return-u32 SSA projection experiment

`pcode-plus-ssa-return-u32-v1` is an experimental input lane for the owned
pure-u32 x86 SysV task. The original full `pcode-plus-ssa-v1` remains unchanged.
This is an explicit low-EAX observation contract, not general register-state or
arbitrary C decompilation support.

The pass accepts only raw P-code and the pinned lifter profile. It first builds
the existing CFG/cell SSA, refuses opaque calls, and roots the slice at low-EAX
exit expressions, every branch condition/target, return target, and every effect
or potentially trapping operation. It closes dependencies through all phi
incoming edges, including loop backedges, then removes unused pure definitions,
phis, liveins and other exit-register observations. CFG blocks/edges and effect
order remain intact. Dead loads, stores and trapping arithmetic are retained;
unknown operations still fail in the original validator. References to missing
definitions reject the projection.

Rendering assigns dense SSA value IDs and removes write/storage annotations;
operand widths, livein register locations, phi predecessors, exit expressions
and effects remain explicit. Source text, ASTs, labels, family IDs and probe
answers never enter this pass. The transformed `Program` is executed by the
bounded audit VM before any model training. Tests additionally cover loops,
diamonds, unused trapping arithmetic, invalid dependencies and refused calls.

## Derive and verify the corpus

The derived corpus copies original hashed artifacts and changes only input
representation, prompt metadata and the corresponding manifest/config. All
1536 pool rows pass projection. The original family quarantine, cross-split
fingerprints, ambiguous-label rejection and duplicate rules run again before
a new release lock is written. Source expressions are not simplified or changed.

```sh
cargo build --locked --release -p fission-forge
python3 scripts/corpus/project_return.py --corpus outputs/corpus-composition-v1 \
  --output outputs/corpus-composition-return-new --forge target/release/forge
target/release/forge corpus-semantics --corpus outputs/corpus-composition-return-new \
  --output outputs/return-semantic-audit-new
target/release/forge tokenizer-train --corpus outputs/corpus-composition-return-new \
  --vocab-size 772 --output outputs/tokenizer-composition-return-new
```

The recorded release preserves the exact ordered **903 train / 130 validation /
188 test** IDs, source labels and raw lifts. The 1221 admitted rows pass **156,288
source comparisons per interpreter**, raw and projected SSA, with zero failures.
This covers 64 probes under register fill states 0 and 0xa5. It is finite-probe
consistency for the declared ABI, not whole-domain equivalence. Test labels are
used only for integrity/IR consistency, never model batches or selection.

Maximum admitted training sequence (input plus target, excluding the final
shifted position) falls from **6094 to 1472 byte positions**. Newly learned
train-only lexical BPE has **500 tokens** and reduces that maximum to **935**
positions, versus 3836 with the previous full-SSA codec. Completion supervision
is still only **56,639 BPE tokens per epoch**; shortening input creates no new
source-label examples. No example is truncated.

## Frozen functional protocol and bounded GPU run

The recorded plan is `configs/evaluation/composition-return-functional-v1.json`.
New builds get a new corpus lock; create a new plan using their real corpus and
codec identities instead of editing recorded experiment identities.

```sh
outputs/eval-venv/bin/python scripts/evaluation/functional.py freeze \
  --corpus outputs/corpus-composition-return-v1 \
  --plan configs/evaluation/composition-return-functional-v1.json \
  --forge target/release/forge --cc /opt/homebrew/opt/llvm/bin/clang \
  --output outputs/composition-return-frozen-new

python3 scripts/colab/run_corpus.py --corpus outputs/corpus-composition-return-v1 \
  --fresh-epochs 8 --fresh-learning-rate 0.0003 \
  --tokenizer outputs/tokenizer-composition-return-v1/tokenizer.json \
  --model-config configs/model/composition-return-rope.json \
  --output outputs/colab-composition-return-new --session forge-return-new --gpu L4

outputs/eval-venv/bin/python scripts/evaluation/score_gpu.py \
  --results outputs/colab-composition-return-new/results \
  --reference-evidence outputs/composition-return-frozen-new \
  --corpus outputs/corpus-composition-return-v1 --output outputs/composition-return-score-new
```

C compilation flags, execution limits, raw-output assessment and the Wasm worker
are unchanged. Evaluator changes only route the declared representation through
prompt compilation/generation, and verify checkpoint/plan lane identity.
The 130 original target sources, reference modules, probes and all **8320**
reference results match the prior RoPE freeze. All generation failures remain
in the denominator; no C prefix, syntax repair, grammar constraint or reference
expression is supplied during generation.

The fresh F32 model retains four layers, hidden size 256, four heads, FF width
1024, context 8192 and RoPE. Its 500-token vocabulary gives **3,406,336 parameters**.
Seed 42, batch 1, LR 0.0003, 7224 updates/eight epochs, full validation and saved
checkpoints every 256 updates remain. A separate longest-row one-update preflight
must pass sample identity and the 85% GPU-memory gate. Lowest full-validation NLL
selects a saved checkpoint before functional scoring. Raw greedy generation
retains the 512-token and 512-byte caps for all 130 validation targets.

This changes the input lane, vocabulary, parameter count and seeded parameter
initialization. It is a measured follow-up, not a single-variable causal ablation.
The acceptance gate remains at least one nonconstant held-out generated function
matching all 64 probes, followed by broader repeatability checks before scaling.
