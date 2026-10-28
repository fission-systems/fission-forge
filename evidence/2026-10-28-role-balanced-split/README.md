# Role-balanced training split: variable binding is learned, new structures are not

The 64-epoch run on the original split copied literals but wrote `a` for `b` in every validation
output, because each training structure occurred with a single assignment of letters (the structure
predicted the letters, so registers never had to be read). This run removes that confound.

## Split (`configs/corpus/shallow-rolebal-v2.json`, `scripts/corpus/shallow.py`)

* Every structure in every split appears with **both** argument assignments (`a` first and `b` first).
* Training: one-operation `add/and/mul/shl` and two-operation `((x - k) & y)`, `((x | k) + y)`, `((x ^ k) * y)`,
  `((x >> k) << 3)`. Validation: one-operation `sub/or/xor/shr` and the outer/inner-reversed two-operation
  structures. Test (not used here): four further structures. Every operator occurs in training (the
  validation operators only as inner operators). `mul` and `shl` share a split because `x*4` and `x<<2`
  are the same behavior and the audit quarantines behavior clones across splits.
* Two construction bugs were fixed before this run: commutative or symmetric structures such as
  `((x - k) + y)` are the same function under a role swap, so the audit removed them as ambiguous prompts
  (the first build left 28 test rows); pairs are now asymmetric and a unit test evaluates both roles.
* Release: **848 train / 224 validation / 112 test**, no quarantined family; 20 training rows dropped as
  ambiguous (`mul`/`shl` by powers of two); semantic audit 151,552 comparisons per interpreter, 0 failures;
  224 validation targets, 14,336 reference cases. L3 prompt lane (decimal literals, digit-isolated BPE).

## Run

Same 3.37M RoPE model, seed 42, LR 0.0003 as before; **64 epochs at batch 8 = 6,784 updates**. Colab L4,
673 s, session terminated, CUs not recorded. Training NLL falls from 1.03 to 0.008; validation NLL is lowest
at step 256 (0.491, selected by the NLL-only rule) and rises to 0.8-0.98.

## Results (224 held-out functions in unseen structures)

| Checkpoint | Compiled | Functions matched | Letters right | Letters swapped | Operators right | Constants right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Selected, step 256 (official)** | 196 | **0** | 105 | 35 | 0 | 12 |
| Final, step 6784 (secondary, CPU generation) | not scored | 0 identical to the label | **178** | 27 | 12 | 99 |
| Previous split, 64 epochs, final (different validation set) | 205 | 0 | 0 | 205 | 204 | 172 |

"Letters right" compares the sequence of `a`/`b` in the output with the label; "swapped" counts outputs
whose sequence is right after exchanging the two letters. The two validation sets differ, so the last row is
context, not a controlled comparison.

On the training split the final checkpoint reproduces 235 of 283 sampled rows exactly (skeleton 275, literals 237).

## Reading

* **Variable binding is learnable once the confound is removed**: 178/224 outputs (79%) assign the letters
  correctly in the final checkpoint versus 0/205 on the confounded split.
* The new validation structures expose a different failure: operators are right in only 12/224 outputs.
  Examples: `(a << 2u)` for `(a >> 2u)`, `(b + 22u)` for `(b - 2u)`, `((b - 8u) + a)` for `((b * 2u) ^ a)`.
  With only eight two-operation and four one-operation training structures the model memorizes whole structures
  instead of reading each operator, so it cannot compose operators it has seen only in other positions. This is
  compositional generalization over a very small structure inventory, a data-coverage limit, and it also lowers
  literal accuracy (99/224).
* The official score is 0/224. No checkpoint is promoted.

## Limits

Single seed; the NLL rule selects a very early checkpoint (step 256); the final-checkpoint rows are CPU
generations scored only by the post-hoc metrics above; the test split was not used; the sample of training
rows is a stride of 3. The earlier confounded-split result and this one use different validation sets.
