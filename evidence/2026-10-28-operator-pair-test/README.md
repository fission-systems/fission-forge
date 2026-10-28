# Operator-pair split: single evaluation of the reserved test split (78/224)

The reserved test split (224 rows, 8 held-out operator pairs, 14,336 reference cases) was evaluated **once**
with the checkpoint that validation NLL had selected before any functional scoring:
`step-00004096` (`training.safetensors` sha256 `aa6cd9ee…aea7a4`, see
[selection](../2026-10-28-operator-pair-split/selection.json)). No other checkpoint, setting or decoding was run
on test; no design change followed from the result.

| | compiled | functionally matched | matched cases |
| --- | ---: | ---: | ---: |
| validation (development set, official GPU run) | 215/216 | 76/216 (35.2%) | 6474/13824 |
| **test (this run)** | **222/224** | **78/224 (34.8%)** | **6152/14336** |

Letters right 195/224, operators right 84/224, constants right 150/224 (validation: 201, 85, 126).

## Per pair (28 rows each, both roles)

| outer∘inner | matched |
| --- | ---: |
| `or∘and` | 26/28 |
| `sub∘and` | 26/28 |
| `sub∘or` | 26/28 |
| `mul∘add`, `mul∘or`, `or∘sub`, `shl∘add`, `shl∘or` | 0/28 each |

Roles are equal (`a`-first 39/112, `b`-first 39/112). Failures: 111 wrong operators with correct letters, 29 wrong
letters, operators and constants, 6 correct operators with a wrong constant. Representative errors:
`(a + 2u) * b` became `(a - 2u) * b`; `(a | 2u) * b` became `(a & 22u) * b`; `(a - 2u) | b` became
`(a + 23u) | b`; `(a + 2u) << 3u` became `(a >> 1u) << 3u`.

## What this says

* **The development-set score did not turn out optimistic here:** 76/216 on validation, 78/224 on test, with the
  checkpoint chosen only by validation NLL.
* **Do not read the closeness as a precise rate.** Results are bimodal per pair (about 93% or 0%), so the
  effective sample is eight pairs, not 224 rows. Validation had four pairs at 43-93% and test has three at 93%; a
  different draw of held-out pairs would move the total by tens of rows. One seed.
* The pairs that work are those whose operators the model has seen composed in many close neighbours
  (`and`, `or`, `xor`, `sub` outer with a bitwise inner). The pairs at 0 again involve a multiply or shift as the
  outer operator, or a subtract as the inner operator under `or`: the same lowering-ambiguity region as the
  validation failures (`x - k` is an add of a negative constant, `x * 3` a `lea`).
* Variable-to-letter binding and constant copying hold on unseen structures (letters 195/224).

## Protocol and provenance

* `scripts/evaluation/functional.py freeze --split test` (new `--split` flag, default `validation`; manifest gains
  `split` and `test_split_sha256`) froze references and probes with the same compiler, bounded Wasm runtime and
  oracle check as validation. Plan: [plan.json](plan.json) (`device` is `cpu`; the other settings equal the
  validation plan). Frozen manifest: [frozen-manifest.json](frozen-manifest.json), sha256 `cf595df6…fd2fe2`.
* Generation used `scripts/evaluation/constant_reproduction.py --split test --max-new-tokens 512` (CPU, greedy,
  KV cache) and scoring `scripts/evaluation/score_external.py --split test`, unchanged output bytes, output cap
  512 bytes (0 truncated). **CPU/GPU parity was verified on validation first:** the step-4096 CPU completions
  equal the official GPU completions for all 216 rows (text compared after whitespace trim), so the CPU run
  stands in for the L4 run. No Colab compute was used.
* Files: [functional-report.json](functional-report.json), [analysis.json](analysis.json),
  [records.jsonl](records.jsonl), [completions-selected-cpu.jsonl](completions-selected-cpu.jsonl),
  [references.json](references.json), [generation-summary.json](generation-summary.json).

## Limits

Single seed, one checkpoint, eight held-out pairs; a synthetic one/two-operation grammar, not real code or
DecBench. The validation split shaped the split construction and the checkpoint choice, but the test split did
not. This test set is now used: further changes to the split design or checkpoint rule should be judged on a
newly drawn split, not on this one. No model is promoted.
