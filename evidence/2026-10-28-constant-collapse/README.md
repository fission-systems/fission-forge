# Why constants collapse: checkpoint does not even reproduce training constants

Checkpoint: compact-lane model, step 10936 (`evidence/2026-10-28-compact-lane`).
`scripts/evaluation/constant_reproduction.py` regenerates **training** rows on CPU
(`forge generate`, greedy, 128 tokens) with every 3rd row of the 1367-row train
split (**456 rows**). Diagnostic only; nothing is scored or selected.

## Measurements

| Check | Result |
| --- | --- |
| Operators + arguments right, constants masked | 418/456 (92%) |
| Whole function text identical to the label | 8/456 (1.8%) |
| Literals identical to the label | 9/456 |
| ...base constants 2..15 | 9/104 |
| ...wide constants 16..255 | **0/352** |
| Distinct generated expressions | 43 |
| Generated literal sets | `7` 223, `13` 87, `4` 60, `7,3` 47, `3` 18, `13,3` 13 |

Literal match by true constant (base rows): 7 matches 5/9 and 13 matches 4/9;
every other constant (2-6, 8-12, 14, 15) matches 0. These two are the model's
fixed outputs, so the matches are coincidence. Constants 2-6, 8 and 9 print
identically in the hex prompt and the decimal output, so a hex-to-decimal
conversion burden does not explain the failure; the model does not copy even
the identity case.

On validation, the same checkpoint's literal is a function of the expression
family only (for example `a + 13u` for all 14 `shallow_one_add_b` inputs), and
the generated value is independent of the true value (7 for 79-93% and 13 for 14-21% of
the rows of each true constant 2..15).

## Loss floor

Average ln(constants per family) is 4.08 nats and a training completion has
40.5 BPE tokens, so a model that never copies the literal has a floor of about
**0.101 nats/token**. Measured mean training NLL over the last 500 updates was
0.139 and validation NLL 0.242. Almost all other loss has been learned; the
remaining training loss is mostly the unlearned literal. NLL therefore cannot
reveal copying unless it falls below this floor.

## Tokenization facts

The learned BPE merges consecutive digits (249 merged tokens contain digits, such
as `13` or `240`), and the prompt's literal is hexadecimal (`#d:4`, two's
complement for subtraction) while the label's literal is decimal. Digit-merged
tokens have no compositional link to their digits, so multi-digit hex-to-decimal
output would be pure lookup. Single-digit constants still failing means the
copy mechanism itself is not learned.

## Conclusion

The collapse is not a validation-only generalization failure: the model already
fails to reproduce constants it saw in training. After 10,936 updates (batch 1,
LR 0.0003) it has learned the expression structure but not the content-based
retrieval of the literal. Whether this is slow optimization, tokenization, or
insufficient inductive support is open; see the proposed experiments in
`docs/literal-copy-experiment.md`. Single checkpoint and seed.
