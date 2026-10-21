# Composition curriculum: expand structures before model size

The bootstrap corpus varied constants within ten fixed C templates. This owned
curriculum generates 96 distinct expression templates and keeps an entire
template family in one split. It is a new, narrower experimental task, **not a
controlled score improvement on the old bootstrap validation or DecBench**.

The target remains raw C: `unsigned int f(unsigned int a, unsigned int b)`.
The grammar composes addition, subtraction, multiplication, bitwise AND/OR/XOR,
literal shifts and unsigned conditional expressions. Constants are small
(2..15); shifts are literal counts 1..7. Arithmetic is defined modulo 2^32.
It does not add pointers, calls, division, general loops or real-project C.
The previous loop/division corpus remains unchanged as a separate challenge.

## Split and admission

A deterministic hash partition stratifies conditional/nonconditional templates
at 25% in each split: 72 train, 12 validation and 12 test families before audit.
Literal variants of a template never cross splits. Every primitive appears in
all three partitions, unlike withholding a primitive exclusively for validation.
Clang-derived canonical AST, erased-literal structure, native probe behavior,
machine bytes and normalized prompt fingerprints pass the existing Rust audit.
No cross-split fingerprint or ambiguous-label policy is relaxed.

On the recorded compiler/lifter, 1536 generated source functions (16 variants of
96 templates) passed **98,304 host-C versus independent Python-oracle cases**.
All 1536 statically lifted and normalized. The audit quarantined three families
and rejected ambiguous or duplicate rows. The frozen release contains:

| Split | Rows / unique C labels | Remaining template families | Completion byte tokens |
| --- | ---: | ---: | ---: |
| Training | 903 / 903 | 65 | 85,877 |
| Validation | 130 / 130 | 11 | 12,154 |
| Test | 188 / 188 | 12 | 17,524 |

Counts describe literal variants, not independent algorithms. Maximum admitted
training input is 6094 byte tokens; no row is truncated. The grammar primitives
remain covered in all released splits after filtering. The follow-up bounded
interpreter audit now compares raw P-code and SSA to the source oracle on every
released row (details below); this is probe consistency, not full x86 equivalence. No benchmark sources or HF rows entered this corpus. Global
external-evaluation overlap remains unmeasured.

## Reproduce data and freeze references

Use the recorded LLVM and pinned Fission/SLA profile; compiler changes produce
a new corpus identity. Choose fresh output directories:

```sh
cargo build --locked -p fission-forge
python3 scripts/corpus/build.py --config configs/corpus/composition-v1.json \
  --output outputs/corpus-composition-new \
  --cc /opt/homebrew/opt/llvm/bin/clang --nm /opt/homebrew/opt/llvm/bin/llvm-nm
```

The committed evaluation plan binds the recorded corpus lock. A freshly rebuilt
corpus can differ in provenance hashes; declare a new plan containing its actual
lock instead of editing recorded evidence. `functional.py freeze` recompiles
all 130 original validation sources to bounded freestanding Wasm and verifies
all **8320 reference cases** before training results are scored.

```sh
outputs/eval-venv/bin/python scripts/evaluation/functional.py freeze \
  --corpus outputs/corpus-composition-v1 \
  --plan configs/evaluation/composition-functional-v1.json \
  --forge target/debug/forge --cc /opt/homebrew/opt/llvm/bin/clang \
  --output outputs/composition-frozen-new
```

## Bounded fresh CUDA experiment

```sh
python3 scripts/colab/run_corpus.py --corpus outputs/corpus-composition-v1 \
  --fresh-epochs 6 --fresh-learning-rate 0.0003 \
  --output outputs/colab-composition-new --session forge-composition-new --gpu A100

outputs/eval-venv/bin/python scripts/evaluation/score_gpu.py \
  --results outputs/colab-composition-new/results \
  --reference-evidence outputs/composition-frozen-new \
  --corpus outputs/corpus-composition-v1 --output outputs/composition-score-new
```

This initializes the 5,380,608-parameter model from scratch (seed 42), rather
than resuming a checkpoint from another corpus. Six epochs are 5418 updates at
batch 1, LR 0.0003, query chunk 256, with full validation and checkpointing every
256 updates plus the final update. A separate one-update preflight deliberately
selects the longest training row. The Python seed selector is checked against
the actual Rust sample ID; memory must remain below 85% of the allocated GPU.

The checkpoint with lowest full-validation NLL among saved snapshots is chosen
**before functional scoring**. All 130 validation prompts are generated on CUDA
with a 512-byte-token limit. No syntax constraints, C repair, wrappers, Markdown
stripping or reference-source insertion are applied. Failed generations remain
in the denominator. The scorer checks source/prompt/checkpoint identity and
rechecks original references using the existing bounded Wasm execution contract.
The test split never enters model batches or checkpoint selection.

The source/helper hashes and job plan travel with the payload. Final and in-job
NLL-best weights are downloaded and verified; other intermediate weight tensors
are omitted from the result archive. The controller releases its named runtime.
Training/inference budgets remain bounded; failed runs retain partial artifacts.

A first functional feasibility signal requires at least one nonconstant held-out
reference matching all 64 probes through unchanged generated C. Compilation
alone, lower NLL, or software-test success does not satisfy that signal. Even a
positive result remains scoped to this small synthetic grammar; it does not
establish real-project or general C decompilation quality or readiness for 20B.

## Recorded byte-codec result

The six-epoch A100 run completed all 5418 updates. NLL-based selection retained
step 1792 (validation NLL 1.381); the final checkpoint regressed to 1.946.
The selected model compiled **0/130** unchanged C outputs, so no candidate reached
execution. The training phase took 770.1 seconds with a sampled whole-device peak
of 10,530 MiB. All raw outputs and the full denominator are retained in
[the experiment record](../evidence/2026-10-22-composition-corpus/README.md).

## Reversible BPE follow-up

The corrective experiment learns a lexical byte-pair codec from **training rows
only**. ASCII identifiers, digits and horizontal whitespace form separate runs;
other bytes are individual segments. Frequency-weighted adjacent pairs merge
within those segments, with deterministic ties. Training prompt weight is 1 and
C-source weight is 8. Special IDs remain PAD/BOS/SEP/EOS = 0/1/2/3. Every byte
retains a base token, so arbitrary output bytes round-trip without UTF-8 repair.
The recorded vocabulary contains 698 tokens (no remaining pair has frequency
at least 2); this is a project codec, not the Hugging Face tokenizer library.

```sh
target/debug/forge tokenizer-train --corpus outputs/corpus-composition-v1 \
  --vocab-size 772 --output outputs/tokenizer-composition-new

python3 scripts/colab/run_corpus.py --corpus outputs/corpus-composition-v1 \
  --fresh-epochs 8 --fresh-learning-rate 0.0003 \
  --tokenizer outputs/tokenizer-composition-new/tokenizer.json \
  --output outputs/colab-composition-bpe-new \
  --session forge-composition-bpe-new --gpu A100
```

The controller adapts the model vocabulary and packages the exact codec. Schema-5
corpus checkpoints embed its specification and training-split hash. Generation
uses the embedded codec; resume rejects changed codec rules. The longest
training row is 3836 BPE positions, compared with 6094 byte positions, and the
GPU preflight selects the actual longest **encoded** row. Train completion tokens
per epoch are 56,639. No row is truncated.

This follow-up changes tokenization, parameter count and update count; it is not
a single-variable causal comparison. Its NLL units differ from byte-codec NLL.
The plan remains seed 42, LR 0.0003, batch 1, full validation every 256 updates,
NLL-only selection before functional scoring, and all 130 validation targets.
The **raw output cap is still 512 bytes**, with at most 512 newly generated
tokens. A whole piece that would exceed the byte budget is not emitted, and the
stop reason is `byte_limit`. The scorer independently reconstructs bytes from
the bound checkpoint pieces. No C prefix, label, grammar constraint, syntax
repair or formatting conversion is supplied to inference.

Use `configs/evaluation/composition-bpe-functional-v1.json` to freeze the BPE
protocol, then the same `score_gpu.py` command with its fresh result/reference
folders. External data and test model batches remain zero. These validation
results inform development; a future final test needs a separately frozen run.

## BPE outcome and position-encoding follow-up

The recorded BPE run compiled **12/130** raw outputs. All twelve reached bounded
execution (768 cases), but all twelve emitted the same C function. Eight cases
matched, and **0/130 functions matched all
64 probes**. The selected step was 1792 (NLL 1.5548 per BPE token); the final
step 7224 regressed to 2.0100. Training took 566.2 seconds on the A100, sampled
whole-device peak 4962 MiB. [Full artifacts](../evidence/2026-10-22-composition-bpe/README.md)
retain all 130 raw outputs and failures. This improves compilability, not proven
semantic decompilation.

The next declared hypothesis replaces learned absolute position embeddings with
[RoPE](https://arxiv.org/abs/2104.09864), retaining the same BPE, corpus, hidden
width, layers, optimizer, update count and functional protocol. Sequence lengths
vary substantially, including where the C completion begins; position encoding
is a hypothesis to test, not an established explanation for the failures.

`configs/model/composition-bpe-rope.json` sets `rotary_positions: true`. RoPE
rotates queries/keys in every layer (split-half convention, theta 10000), using
primitive differentiable operations and precomputed tables. Values stay
unrotated; incremental cache offsets match full-prefix positions. The old
learned-position mode remains the default for historical configurations. The
parameter count falls to **3,507,712** after removing the 8192×256 table. Changing
the parameter set also changes parts of seeded initialization; this is an
architecture comparison, not a one-variable proof of causality.

CPU tests check an independent trigonometric formula, relative dot-product
invariance, causal masking, tiled/full logits and gradients, detached validation
and cached/full-prefix logits for both position modes. A real corpus update and
resume are checked separately. A GPU run must still pass its longest-row
memory preflight before training.

```sh
python3 scripts/colab/run_corpus.py --corpus outputs/corpus-composition-v1 \
  --fresh-epochs 8 --fresh-learning-rate 0.0003 \
  --tokenizer outputs/tokenizer-composition-bpe-v1/tokenizer.json \
  --model-config configs/model/composition-bpe-rope.json \
  --output outputs/colab-composition-rope-new \
  --session forge-composition-rope-new --gpu A100
```

Freeze `configs/evaluation/composition-bpe-rope-functional-v1.json` and score all
130 targets through the same unchanged bounded Wasm evaluator. The explicit
model config, codec, train split and full reference set are hash-bound. No test
model batches, syntax constraints, source repair or C-prefix insertion are added.

## Recorded RoPE result

The L4 run completed all 7224 updates. It selected step 768 at validation NLL
0.5314; the final NLL was 0.7361. Training-command wall time was 616.1 seconds,
sampled whole-device peak 4874 MiB. Raw C compiled **16/130**, but **0/130 matched
all 64 probes**. The sixteen compilable outputs contained only two distinct C
functions; the most frequent repeated fourteen times. Four of 8320 total cases
matched. [Complete record](../evidence/2026-10-22-composition-rope/README.md).

| Trial | Parameters | Selected validation NLL | Compiled raw C | All-case matches |
| --- | ---: | ---: | ---: | ---: |
| Byte, A100, six epochs | 5,380,608 | 1.3810 / byte | 0/130 | 0/130 |
| BPE, A100, eight epochs | 5,604,864 | 1.5548 / BPE token | 12/130 | 0/130 |
| BPE + RoPE, L4, eight epochs | 3,507,712 | 0.5314 / BPE token | 16/130 | 0/130 |

Byte/BPE NLL units differ. Architecture, parameter set, some initialization and
hardware also differ; this table is not a causal ablation or a DecBench score.
All outputs are preserved unchanged, with failures retained in the denominator.
The functional-success gate remains **unmet**. Larger-model training is deferred
until input/label fidelity, syntax and semantic conditioning are investigated;
repeating the current small corpus does not supply new supervision.


## Bounded IR execution consistency

`forge corpus-semantics` first verifies the existing corpus release integrity,
then executes both raw P-code and its normalized SSA for all 1221 released rows.
The ABI is explicit: x86 SysV RDI/RSI contain zero-extended u32 inputs, RSP is
0x1000, the mapped return slot contains 0xfeedbaad, and low EAX is the result.
Each of the 64 source-oracle probes runs with other register bytes initialized
to 0 and 0xa5. Both return value and return target are checked. Unmapped RAM
reads, opaque calls, unsupported operations and exhausted fuel reject the case.
This VM shares arithmetic helpers between raw and SSA paths; the independently
computed Python oracle and earlier native-C checks supply the source leg.

```sh
cargo build --locked --release -p fission-forge
target/release/forge corpus-semantics --corpus outputs/corpus-composition-v1 \
  --output outputs/composition-semantic-audit-new
```

The recorded run passed **156,288 comparisons per interpreter**, with zero
mismatches. All train/validation/test rows are included for IR consistency only;
no test label enters model inference, optimization or checkpoint selection.
The result is restricted to these pure-u32 functions and finite probes. It does
not validate general memory, arbitrary x86 instructions, calls, or whole-domain
semantic equivalence. Full per-row evidence is retained in the experiment record.

## Frozen-checkpoint input-conditioning control

`forge corpus-diagnose` measures teacher-forced completion NLL on all 130
validation targets, once using each target's own input and once replacing only
the input with row `(i + 17) % 130`. Donors must belong to a different template
family. The entire target C and EOS remain identical between arms. Header and
expression-body-plus-EOS losses are reported separately, with token-weighted
aggregates and per-row sums. It performs zero optimizer updates and uses no
test model examples. It does not generate C or establish functional accuracy.

```sh
python3 scripts/colab/run_corpus.py --corpus outputs/corpus-composition-v1 \
  --diagnostic-checkpoint outputs/colab-composition-bpe-rope-l4-v1/results/fresh/checkpoints/step-00000768 \
  --output outputs/colab-conditioning-new --session forge-conditioning-new --gpu L4
```

The controller packages the embedded checkpoint model and codec, verifies state
and weight hashes before evaluation, and verifies that weights remain unchanged.
The diagnostic phase has a 300-second limit inside the existing bounded session
lifecycle. A small wrong-minus-correct loss gap would suggest weak conditioning
under this particular control; it would not identify the cause by itself.


The recorded L4 control gave full-completion NLL **0.531415 → 0.609941** when
swapping inputs; body-plus-EOS NLL was **0.869144 → 0.998437**. The common header
NLL was approximately 0.009017 in both arms. Own-input total NLL was lower on
76/130 rows. This shows some conditioning under the declared control, alongside
substantial common-header predictability. It does not establish semantic
generation: previous all-case matches remain **0/130**. No updates were run.
See [the complete audit and diagnostic record](../evidence/2026-10-22-conditioning/README.md)
for all row results, caveats, resource use and hashes. The next representation
experiment should compact return/control dependencies without using labels,
recheck transformed-prompt collisions and retain the existing functional gate.
