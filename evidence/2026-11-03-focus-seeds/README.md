# Focus split, seeds 43 and 44: the 53/244 does not reproduce as a stable gain

Repeats the focus-split experiment ([2026-10-29](../2026-10-29-focus-split/README.md), seed 42) with **two more training
seeds per arm**, everything else identical: same corpus locks, tokenizers, model configs, frozen references (the
seed-42 `outputs/focus-{uniform,weighted}-frozen` are reused), 64 epochs, batch 8, LR 0.0003, L4, NLL-only checkpoint
selection, 512 new tokens, GPU generation of the NLL-selected checkpoint and the unchanged `score_gpu.py` scorer. The
seed controls weight initialization and sampler order (`scripts/colab/run_corpus.py --fresh-seed`, new in `00b8390`;
the job file differs from seed 42 only by `"seed"`). Seeds 43 and 44 were fixed before any result was seen. The test
split (216 rows) is still unused.

Auxiliary, as before: the final checkpoint of every run generated on CPU with `forge generate` and scored with
`score_external.py`. Intermediate checkpoints keep no weights, so only the selected and final steps can be scored.
Run times 965-1037 s.

## Result (validation, 244 targets, functionally matched)

| arm | seed | NLL-selected step | **official (NLL-selected)** | final (auxiliary) |
| --- | ---: | ---: | ---: | ---: |
| uniform | 42 | 512 | 0 | 28 |
| uniform | 43 | 512 | 1 | 36 |
| uniform | 44 | 6400 | 0 | 0 |
| weighted | 42 | 13824 | 53 | 49 |
| weighted | 43 | 768 | 0 | 28 |
| weighted | 44 | 8192 | 39 | 24 |

Per pair, matched rows of 24-28 (`pair-table.json` has all twelve columns). Selected / final for each seed 42, 43, 44:

| pair | uniform selected | uniform final | weighted selected | weighted final |
| --- | --- | --- | --- | --- |
| `or∘sub` | 0, 0, 0 | 28, 28, **0** | 28, **0**, 28 | 26, 28, 24 |
| `add∘shl` | 0, 0, 0 | 0, 7, 0 | 24, 0, 4 | 20, 0, 0 |
| `mul∘or` | 0, 0, 0 | 0, 0, 0 | 0, 0, 7 | 0, 0, 0 |
| `shl∘add` | 0, 1, 0 | 0, 0, 0 | 1, 0, 0 | 3, 0, 0 |
| `add∘mul` | 0, 0, 0 | 0, 1, 0 | 0, 0, 0 | 0, 0, 0 |
| `mul∘add`, `shl∘or`, `shr∘add`, `shr∘sub` | 0 | 0 | 0 | 0 |

## What this says

* **`or∘sub` is reproducible, not guaranteed.** The final checkpoint solves it (24-28 of 28) in five of six runs, in
  both arms, so extra pair coverage plausibly explains it; uniform seed 44 stayed at 0 even at the end. With one
  failure in six runs the data cannot rank the two arms.
* **The `add∘shl` result of seed 42 did not reproduce.** Weighted solved it only in seed 42 (24/24 selected, 20/24
  final), then 0 and 0-4 in seeds 43 and 44, while the uniform arm got 7/24 once. The earlier suggestion that
  oversampling the weak pairs causes this is **not supported**. `mul∘or` (7/28, once) and `add∘mul`/`shl∘add` (1-3
  rows) are noise-level; `mul∘add`, `shl∘or`, `shr∘add`, `shr∘sub` stayed at 0 in all twelve scored checkpoints.
* **Run-to-run spread is as large as the claimed effect.** The same arm and data score 0 to 53 (weighted, official)
  and 0 to 36 (uniform, final) depending only on the seed. Aggregates are driven by whether the one or two solvable
  pairs flip, so the effective sample is small: a handful of pairs times three seeds, not 244 rows.
* **Weighted looks better on the official number (53, 0, 39 vs 0, 1, 0) but that is mostly the selection rule.** At the
  final checkpoint the ranges overlap (49, 28, 24 vs 28, 36, 0; means 33.7 and 21.3 from three runs). The weighted arm
  also has 13.5% more rows and 13% more updates. Not a demonstrated advantage.
* **NLL checkpoint selection is not a usable model-selection rule on these pairs.** It chose step 512 (uniform 42, 43),
  768 (weighted 43) and mid-run steps otherwise; in three of six runs the selected checkpoint scores 0-1 while the final
  one scores 28-49. Validation NLL runs 0.17-0.22 at its minimum (seeds 43-44) and 0.20-0.34 at the end (seeds 43-44), with large step-to-step
  swings (`training-nll.json`, seeds 43-44; seed 42 in the earlier evidence). The protocol stayed unchanged here so the
  numbers are comparable, but any future arm comparison needs a selection rule fixed in advance that does not depend
  on validation NLL alone (for example a predeclared final step, or a separate small functional probe set that is not
  the validation split).

## Limits

Three seeds per arm; validation pairs were chosen because they scored 0 earlier, so none of these numbers estimate
generalization; synthetic grammar, 3.37M-parameter model; no model promoted; failure categories (operator confusion)
were not re-analyzed for the new seeds.

Files: `pair-table.json`, `training-nll.json`, `selection-*-s{43,44}.json`, `job-*-s{43,44}.json`,
`functional-report-{selected,final}-*-s{43,44}.json`, `completions-{selected,final-cpu}-*-s{43,44}.jsonl`;
`scripts/evaluation/pair_table.py` builds the pair table from the scorer output folders.
