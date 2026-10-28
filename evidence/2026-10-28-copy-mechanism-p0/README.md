# P0/P1: can a from-scratch model learn to copy a literal?

Question (from `docs/literal-copy-experiment.md`): earlier scratch runs never copied
constants, even from training rows. Is that a task/representation limit or an
optimization/training-length limit? This isolates the mechanism on the simplest possible task.

## Task (diagnostic only, not a benchmark)

One template, `unsigned int f(unsigned int a, unsigned int b) { return (a + k); }`, in all three
splits with **disjoint literals** k drawn from a seeded partition of 2..255 (train 180 /
validation 34 / test 34; the test split is never used). The audit's cross-split shape rule is
skipped only through the explicit `diagnostic_shared_shapes` config flag; `audit.json` records the
scope `constant_interpolation_diagnostic_shapes_shared_across_splits`. Prompts use the L3 setup
(compact lane with decimal literals, digit-isolated BPE: vocabulary 321, 37.6 completion tokens
on average). Same 3.37M RoPE model family as the main experiments (`configs/model/p0-shallow-rope.json`),
seed 42, CPU, `forge train-corpus`. Lock `79ef5971...`; semantic audit 0 failures; references 34 targets /
2176 cases. Checkpoint = lowest validation NLL (also reported: final step). Validation literals
are the held-out constants; train literals are a stride-6 sample (30 rows) of the training split.

## Results

| Run | Batch | Updates | Epochs | LR | Selected step | Held-out literals (selected) | Train literals (selected) | Functions matched (selected) | Held-out literals (final) | Functions matched (final) | Final val NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1-b1-1440-lr3e-4 | 1 | 1440 | 8 | 0.0003 | 1440 | **0/34** | 0/30 | 0/34 | 0/34 | 0/34 | 0.1274 |
| r2-b1-7200-lr3e-4 | 1 | 7200 | 40 | 0.0003 | 7200 | **33/34** | 29/30 | 33/34 | 33/34 | 33/34 | 0.0023 |
| r3-b8-1440-lr3e-4 | 8 | 1440 | 64 | 0.0003 | 1440 | **34/34** | 30/30 | 34/34 | 34/34 | 34/34 | 0.0011 |
| r4-b8-1440-lr1e-3 | 8 | 1440 | 64 | 0.001 | 1320 | **22/34** | 20/30 | 22/34 | 24/34 | 24/34 | 0.0312 |
| r5-b1-1440-lr1e-3 | 1 | 1440 | 8 | 0.001 | 1440 | **0/34** | 2/30 | 0/34 | 0/34 | 0/34 | 0.153 |
| r6-b8-180-lr3e-4 | 8 | 180 | 8 | 0.0003 | 180 | **1/34** | 0/30 | 1/34 | 1/34 | 1/34 | 0.1756 |
| r7-b8-360-lr3e-4 | 8 | 360 | 16 | 0.0003 | 330 | **2/34** | 1/30 | 2/34 | 0/34 | 0/34 | 0.1448 |

(Counts of 34/34 and 33/34 are held-out literals the model never saw in training; "functions
matched" is the unchanged frozen functional scorer on 2176 cases.)

Validation NLL of R2 stays near 0.14 (the "literal never copied" floor) until about step 2400,
then falls to 0.0023 by step 7200 (0.0338 at 4200, 0.0133 at 4800): the copy mechanism appears
partway through training.

## Reading

* A 3.4M scratch model **can** learn to copy a held-out literal, including numbers it never saw in
  training. The earlier failures were not an inherent limit of the architecture or of decimal
  rendering.
* What predicts success here is the number of **passes over the data**, not the batch size or
  update count alone: 8 and 16 epochs fail (R1, R5, R6, R7) at both batch 1 and batch 8; 40 epochs
  (R2) and 64 epochs (R3, R4) succeed. R3 reached it in 1,440 updates at batch 8 while R1 failed with
  1,440 updates at batch 1, so updates are not the variable either. **This corrects an earlier
  statement in this session that batch 8 was the bottleneck; R6/R7 refute it.**
* LR 0.001 hurts at batch 8 (22/34) and does nothing at batch 1 within 8 epochs (R5).
* The main-task scratch runs all used 8 epochs, which is below the range where copying appeared
  here. Their failures may therefore be undertraining in epochs rather than a representation
  problem. This is a hypothesis for the main task, not a result: the main corpus has many templates
  and the transition point may differ.

## Limits and provenance

Single seed and one model size; a one-template task; the transition epoch is only bracketed (16-40).
R1-R3 were first run before an accidental deletion of `outputs/`; the rerun reproduced their
held-out counts exactly (0, 33, 34) and the identical corpus lock and reference manifest hash
(`19fc2620...`). R4-R7 ran only after the rebuild. No GPU or CU was used (CPU, about 0.05 s per
update at batch 1).
