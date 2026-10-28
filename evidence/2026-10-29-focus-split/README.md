# Focus split: training design for the nine pairs that scored 0 (53/244 vs 0/252)

The operator-pair split left nine held-out pairs at 0 (`add∘mul`, `add∘shl`, `shr∘add`, `shr∘sub` on validation;
`mul∘add`, `mul∘or`, `or∘sub`, `shl∘add`, `shl∘or` on test): the outer operator is mul/shl or the inner operator is
add/sub/or, the operators with the fewest training pairs. This experiment keeps **exactly those nine pairs as the
validation pairs**, so the old 0 is a same-pair baseline, and changes the training side.

## Design (`scripts/corpus/shallow.py`: `rf_*`, `scripts/corpus/rf_config.py`)

* Validation: the nine pairs above (244 rows after ambiguity drops, 15,616 reference cases). Test: eight **fresh**
  pairs (216 rows; `and∘shr`, `and∘sub`, `mul∘xor`, `shl∘xor`, `xor∘add`, `xor∘and`, `xor∘mul`, `xor∘shl`;
  `and∘sub` and `xor∘add` were validation pairs of the earlier split, so they are not new to the project). **Test
  not used here.**
* Training pairs: 34. Every operator is the outer operator of at least three training pairs (shr two: only four
  shr-outer pairs exist) and the inner operator of at least three. In the earlier split `add`/`or` were inner
  operators of only two training pairs.
* Arm **uniform** (control): 8 extra constant variants for every training structure, as before. 1781 train rows.
* Arm **weighted**: 16 extra variants for the 15 training pairs of the weak kind (outer mul/shl or inner
  add/sub/or), 8 for the rest. 2021 train rows (+13.5%), 16,168 updates versus 14,248.
* Everything else unchanged: 3.37M RoPE model, seed 42, LR 0.0003, 64 epochs, batch 8, L3 prompt lane, digit-isolated
  BPE (346/347 tokens), L4, NLL-only checkpoint selection. No quarantine; semantic audit 286,848 / 317,568 comparisons
  per interpreter, 0 failures. Run time 960 s and 1027 s.

## Result (validation, 244 targets)

| arm / checkpoint | compiled | functionally matched | matched cases |
| --- | ---: | ---: | ---: |
| earlier split, same nine pairs, NLL-selected | – | 0/252 | – |
| uniform, NLL-selected (step 512) — official | 194 | **0/244** | 611 |
| uniform, final (step 14248) — auxiliary | 243 | 28/244 | 3353 |
| weighted, NLL-selected (step 13824) — official | 223 | **53/244 (21.7%)** | 4591 |
| weighted, final (step 16168) — auxiliary | 244 | 49/244 | 4589 |

Per pair (cases of 24-28 rows; selected / final):

| pair | earlier | uniform | weighted |
| --- | --- | --- | --- |
| `or∘sub` | 0/28 | 0 / **28** | **28** / 26 |
| `add∘shl` | 0/24 | 0 / 0 | **24** / 20 |
| `shl∘add` | 0/28 | 0 / 0 | 1 / 3 |
| `add∘mul`, `mul∘add`, `mul∘or`, `shl∘or`, `shr∘add`, `shr∘sub` | 0 | 0 / 0 | 0 / 0 |

## What this says

* **Better pair coverage plus weighting moved two of nine pairs from 0 to about 90-100%** (`or∘sub`, `add∘shl`) and
  one to a few rows; six pairs stay at 0. The aggregate is two pair-sized steps, not a smooth gain.
* **`or∘sub` is explained by coverage alone:** the uniform arm solves it too (final), the earlier split could not.
  **`add∘shl` appears only in the weighted arm**; with one seed and 13% more updates and rows in that arm, this
  cannot be separated from run-to-run variation or from the extra compute. The oversampling effect is suggestive,
  not established.
* **Validation NLL does not track functional accuracy on these pairs.** It bottoms out early (uniform: 0.215 at
  step 512, then 0.23-0.35) while accuracy appears only late; the predeclared NLL rule picked a step-512 uniform
  checkpoint that fails 50 compilations and matches nothing, so the official uniform number is 0 while its final
  checkpoint matches 28. The weighted selection (step 13824) happened to land late. Treat the official uniform 0
  as a checkpoint-selection failure, not as the arm's capability. Intermediate checkpoints kept only `state.json`
  in the downloaded archive (weights for the selected and final steps only), so other steps cannot be scored.
* **Remaining failures are operator confusions in the lowering-ambiguous region:** `(a + 2u) * b` becomes
  `(a - 2u) * b` (add/sub differ only by a constant's sign), `(a + 2u) >> 3u` becomes `(a * 20u) >> 3u`, `(a * 2u) + b`
  becomes `a * 3u`. Final-checkpoint failures: weighted 145 wrong operators with correct letters, 38 wrong
  letters and operators, 12 correct operators with a wrong constant; uniform 177 and 39.

## Limits

Single seed per arm; validation pairs were chosen because they failed before, so the 53 is a development-set
result on a deliberately hard set, not an estimate. Weighted differs from uniform in rows and updates as well as
weighting. The 216-row test split is untouched. Synthetic grammar only; no model is promoted.

Files: `functional-report-*.json`, `completions-*.jsonl`, `pair-table.json`, `training-nll.json` (validation NLL by
step), `selection-*.json`, `corpus-lock-*.json`, `tokenizer-*.json`, `ir-semantics-report-*.json`,
`frozen-manifest-*.json`. Final-checkpoint rows are CPU generations scored with `score_external.py`.
