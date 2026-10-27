# Train-only wide-literal variants: constant copying not established

Hypothesis: the 3.37M shallow model ignored operand literals because every
training template used only constants 2..15 (14 variants per template). The
balanced follow-up emitted one sentence for validation inputs with different
constants, for example `a + 12u` for constants 2, 3 and 4.

Change (one data axis, validation/test untouched): `configs/corpus/shallow-wide-v3.json`
adds `extra_variants` to the 24 training families. Added literals are distinct,
deterministic and outside 2..15 (2..255; shifts 1..31, hence 17 for the one-op
shift families). `shallow.instantiate` refuses index >= 14 for any non-train
family. The Rust audit now accepts `variants_per_family + extra_variants[family]`;
`extra_variants` must name explicit train families. No leakage/ambiguity rule is
weakened.

## Data

1762 source functions, 112,768 native-C/oracle comparisons passed, all lifted.
Release: **1367 train / 205 validation / 112 test** (balanced-v2: 317 / 205 / 112).
`validation.jsonl` and `test.jsonl` are byte-identical to balanced-v2
(SHA-256 d25bd78d... and 3786a157...). One previous training row,
`shallow_one_mul_a_0000-O1`, is now excluded as `ambiguous_prompt_labels`
(an added multiplier of 16 compiles like `shl 4`); the rule was not relaxed.
All 205 references and 13,120 reference cases re-froze identically.
Train-only BPE: 692 tokens (balanced-v2: 424).

## Run

Fresh 3.37M RoPE model (`configs/model/shallow-rope.json`), seed 42, LR 0.0003,
batch 1, **eight epochs / 10,936 updates**, L4. Training wall time 321.9 s, sampled
whole-device peak 490 MiB, total job 773 s; the session terminated (no active
Colab sessions afterwards). Account CUs were not recorded for this run.
NLL-only selection chose **step 1280, NLL 0.2573** (balanced-v2: 0.2259).
Validation NLL stayed 0.26-0.44 after epoch one and rose in later epochs.

Execution used the working tree on top of 97e284f with uncommitted edits to
`scripts/corpus/{build,shallow,test_shallow}.py` and `crates/forge/src/corpus.rs`.
Rust tests (30) and corpus tests pass; model math is unchanged.

## Outcome (205 validation functions, 13,120 reference cases)

| Run | Compiled | All-case matches | Case matches | Distinct outputs |
| --- | ---: | ---: | ---: | ---: |
| balanced-v2 (317 train rows) | 201 | 0 | 635 | not recomputed |
| wide-v3 (1367 train rows) | 188 | **1** | 904 | 18 |

Post-hoc analysis of the raw outputs: constants match the reference expression
in **15/205**, operators in 87/205 and argument variables in 53/205 (multiset
comparison on the returned expression). In 9 of the 15 validation families, all
variants (different constants) get one identical output (for example `a - 7u`
for all 14 `shallow_one_add_b` inputs, `a & 7u` for `shallow_one_xor_b`), so the
model is still not reading the literal. The single match,
`shallow_two_sub_b_0005-O1`, is one of 205 and is treated as coincidental until
shown otherwise; the other 17 syntax failures stay in the denominator.

## Conclusion

The wide-literal data hypothesis is **not supported**: more varied constants did
not produce input-conditioned output, and validation NLL did not improve. No
checkpoint is promoted. The positive result from the prior curriculum (8/8
training-example memorization) is unchanged. Data, codec size and update count
changed together, so this is not a single-variable ablation. The original nested
130-function task, reserved test split and DecBench were not measured.

Next candidates, none yet run: shorten the informative span (a compact prompt
that keeps only the operator, argument register and literal), add an explicit
auxiliary copy objective, or test a higher LR on the diagnostic set where
memorization succeeded (LR 0.003 vs the 0.0003 used here).

## LR 0.003 re-run on the same corpus (worse)

Same wide-v3 corpus, tokenizer, 3.37M RoPE model, seed, batch 1 and 10,936 updates;
only the learning rate changes from 0.0003 to **0.003**, the value used by the
eight-example memorization diagnostic. `scripts/colab/run_corpus.py` previously
rejected fresh-job LR above 0.001; its client-side bound was raised to 0.003 (the
driver and Rust trainer had no such bound). The first attempt was rejected before
any session started, so it consumed no GPU. A new plan,
`configs/evaluation/shallow-wide-lr3e-3-functional-v3.json`, was frozen to a
separate directory (205 targets, 13,120 reference cases). L4 job took 800 s;
the session terminated. Account CUs were not recorded.

Training did not diverge: mean train NLL fell from 1.812 (first 200 updates) to
0.289 (last 500). Validation NLL was erratic (1.3, 2.3, 1.4, 0.76, 1.3, ...) and
the NLL-only selection chose **step 10496, NLL 0.3262**, worse than both the
0.0003 run (0.2573) and balanced-v2 (0.2259).

| Run | Compiled | All-case matches | Case matches | Distinct outputs |
| --- | ---: | ---: | ---: | ---: |
| wide-v3, LR 0.0003 | 188 | 1 | 904 | 18 |
| wide-v3, LR 0.003 | **86** | **0** | 230 | 18 |

Compile failures rise to 119/205 and no function matches. Constants agree with
the reference in 7/205 outputs (15 at LR 0.0003), operators in 56/205 and
argument variables in 72/205; 8 of 15 validation families still emit one
output for all constants. Higher LR therefore does not make the model read the
literal and harms syntax acquisition. The memorization diagnostic's LR does not
transfer to this corpus. No checkpoint is promoted; the remaining untested
candidates are a compact prompt and an explicit auxiliary copy objective.
