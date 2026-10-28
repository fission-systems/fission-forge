# L3 lane, 64 epochs, batch 8: the scratch model copies literals; the failure is argument binding

Hypothesis from `2026-10-28-copy-mechanism-p0`: earlier scratch failures were too few passes over
the data (all used 8 epochs), not the representation. This re-runs the L3 setup (compact lane,
decimal literals, digit-isolated BPE) with the same rows, model, seed and LR as the 8-epoch
run, changing only the optimization: **64 epochs at batch 8 = 10,936 updates** (same update count).
Colab L4, wall 773 s, session terminated; CUs not recorded. Corpus rebuilt on 2026-10-28 after an
accidental deletion of `outputs/`: release files and tokenizer are byte-identical to the earlier
run; only manifest provenance differs, so the evaluation plan is bound to the rebuilt lock digest.
`score_gpu.py` check "steps = epochs x train rows" was generalized to include the batch size.

## Training signal

Mean training NLL falls from 1.04 (first 200 updates) to **0.010** (last 500), far below the
~0.10 floor of a model that never copies a literal. Validation NLL is lowest at step 2048
(0.286, selected by the NLL-only rule) and rises to 0.44-0.54 later.

## Training-row reproduction (456 rows, CPU generation)

| Checkpoint | Skeleton right | Whole text identical | Literals right | Literals above 15 right |
| --- | ---: | ---: | ---: | ---: |
| 8 epochs, batch 1 (earlier run, final) | 382 | 8 | 10 | 2/352 |
| 64 epochs, batch 8, step 2048 (selected) | 449 | 251 | 252 | 180/352 |
| 64 epochs, batch 8, step 10936 (final) | **456** | **399** | **399** | **308/352** |

## Validation (205 held-out functions, 13,120 cases)

| Run | Compiled | Functions matched | Case matches | Operators right | Constants right |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 epochs, batch 1 (earlier, selected) | 152 | 0 | 288 | 78 | 14 |
| **64 epochs, batch 8, selected (step 2048)** | **205** | **0** | **1991** | **204** | **114** |
| 64 epochs, batch 8, final (secondary) | 205 | 0 | not scored with GPU report | 204 | **172** |

The functional score is 0/205 for both arms. The failure is one systematic thing: in all 205
outputs the model writes `a` where the reference writes `b` and vice versa (`(a + 2u)` for
`(b + 2u)`; `((a ^ 2u) + b)` for `((b ^ 2u) + a)`). A **post-hoc** check that swaps the letters
`a` and `b` in the output and compares the returned expression gives an exact match in **114/205
(selected) and 172/205 (final)**. This analysis is not part of the frozen protocol and does not change
the reported 0/205.

## Why the binding fails

Validation templates are the training templates with the roles exchanged (the role holdout).
In training, every operator structure occurs with exactly one assignment of letters, so the
structure alone predicts the letters and the register names (`r38` is `a`, `r30` is `b`) carry no
information the model needs. It learned structure-to-letters, not register-to-letter. The
benchmark design is therefore what makes this binding unlearnable from the training data; it is a
data confound, not a capacity limit. (The fine-tuned Qwen model handled two-operation roles but
failed one-operation functions for the opposite structural reason; see `2026-10-28-pretrained-finetune`.)

## Conclusion and limits

With enough passes over the data a scratch 3.4M model reads operators (204/205), copies held-out
literals (172/205 final) and reproduces training rows (399/456), so the earlier operator, copy and
rendering failures were undertraining plus the 8-epoch budget. The remaining failure is argument
binding caused by the role-confounded split. Single seed; checkpoint chosen by validation NLL
reached only step 2048, while the (secondary) final checkpoint is much better on every copy metric;
only validation was scored; no checkpoint is promoted. A representation control (compact lane
with hex literals under the same 64-epoch regime) is running to separate the effect of decimal
literals and digit isolation from training length.
