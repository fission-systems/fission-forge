# Operator-pair split: held-out operator combinations partly generalize (76/216)

Previous splits failed for two different reasons: the confounded split made structure predict the
letters (0/205), and the small role-balanced split let the model memorize its 12 training structures
(operators right in 12/224). This split gives the model **many operator pairs** with both argument roles
and holds out whole pairs, so a validation function is a *new combination of operators that each occur
in training as both an outer and an inner operator*.

## Split (`configs/corpus/shallow-opairs-v1.json`, `scripts/corpus/shallow.py`)

* 53 usable (outer, inner) pairs; **37 trained, 8 validation, 8 test** (test not used here). Every structure
  of every split appears with both `a`-first and `b`-first assignments. One-operation functions for all eight
  operators and both roles are in training only.
* Every operator is the outer operator of at least two and the inner operator of at least two training pairs.
* Pairs are excluded when a role swap yields the same function (`(x - k) + y`), a degenerate function
  (`(x & k) >> 3`, always 0 for small k) or a behavior clone of another structure (`(x | k) >> 3`, `(x ^ k) >> 3`
  equal `x >> 3` for k < 8; the audit quarantined six families before this rule). `mul` and `shl` share a
  split. A build-time native-check timeout was raised from 180 to 900 s for the larger source set.
* Release: **1812 train / 216 validation / 224 test**, nothing quarantined, 88 rows dropped as ambiguous
  (`mul`/`shl` by powers of two); semantic audit 288,256 comparisons per interpreter, 0 failures; 216
  validation targets, 13,824 reference cases. L3 prompt lane (decimal literals, digit-isolated BPE, 346 tokens).
* Held-out validation pairs (outer, inner): `add∘mul`, `add∘shl`, `and∘add`, `and∘sub`, `shr∘add`, `shr∘sub`,
  `xor∘add`, `xor∘or`.

## Run

Same 3.37M RoPE model, seed 42, LR 0.0003; **64 epochs at batch 8 = 14,496 updates**. Colab L4, 931 s, session
terminated, CUs not recorded. Training NLL 1.04 to 0.006. Validation NLL stays at 0.15-0.21 (it rose to 0.8-0.98
on the role-balanced split); the NLL-only rule selects **step 4096** (0.151).

## Results (216 held-out functions, 13,824 cases)

| Checkpoint | Compiled | Functions matched | Case matches | Letters right | Operators right | Constants right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **NLL-selected, step 4096 (official)** | 215 | **76 (35.2%)** | 6474 | 201 | 85 | 126 |
| Final, step 14496 (secondary, CPU generation, same scorer) | 215 | 39 | 4520 | 198 | 40 | 132 |

Functions matched by held-out pair (official / final):

| Pair | Selected | Final |
| --- | ---: | ---: |
| `xor∘or` | **26/28** | 27/28 |
| `and∘add` | 22/28 | 12/28 |
| `xor∘add` | 16/28 | 0/28 |
| `and∘sub` | 12/28 | 0/28 |
| `add∘mul`, `add∘shl`, `shr∘add`, `shr∘sub` | 0 | 0 |

On training rows the final checkpoint reproduces 404 of 453 sampled rows exactly (skeleton 442).

## Reading

* A scratch 3.4M model **generalizes to unseen operator combinations**: 4 of 8 held-out pairs reach 43-93% and
  letters are right in 201/216 outputs, so register binding and literal copying hold on new structures. This is
  the first non-trivial held-out functional score of a from-scratch model in this project (previous splits: 0).
* The NLL-selected checkpoint (step 4096) is much better than the final one (76 vs 39): later training memorizes
  training pairs. Selecting by validation NLL, as predeclared, worked here.
* The four failing pairs all combine a shift or multiply with an add/sub, and the errors are mostly the wrong
  inner or outer operator: `(a - 2u) & b` becomes `(a + 22u) & b`; `(a + 2u) >> 3u` and `(a - 2u) >> 3u` both become
  `(a * 3u) >> 3u`; `(a << 4u) + b` becomes `(a * 9u) + b`. Among non-matching outputs: 66 wrong operators and
  constant, 50 wrong operator only, 15 operators, constant and letters, 9 constant only. The compiler lowers
  `x - k` to an add of a negative constant and `x * 3` to a `lea` (an add of shifted copies), so these inner
  operators have similar P-code and are the least exposed in training. This is a limited operator inventory and
  lowering-ambiguity effect, not evidence of a capacity limit.

## Limits

Single seed; validation was used to iterate on the split design (several construction fixes, see the commit log)
and to choose the checkpoint, so 35.2% is a development-set result, not an unbiased estimate; the reserved
the test split was evaluated once afterwards (78/224, see `../2026-10-28-operator-pair-test/`); the final-checkpoint rows are CPU
generations scored with the same functional scorer; no model is promoted; this is a synthetic one/two-operation
grammar, not real code or DecBench.
