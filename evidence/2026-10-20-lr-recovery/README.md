# Explicit lower-LR recovery: preserve best NLL, reject regressed final

A separate Colab A100 experiment starts from the historical **step-120**
validation-NLL best checkpoint. It retains weights, both Adam moments,
optimizer step and sampler, explicitly changes LR **0.001 → 0.0001**, and takes
**960 additional updates** to step 1080. It does not resume the failed step-2400
checkpoint or represent a single continuous 3360-step lineage.

| Measurement | Value |
| --- | --- |
| Parameters / context | 5,380,608 / 8192 |
| Initial validation NLL, step 120 | 2.036877629 |
| Best validation NLL, step 240 | **1.965501113** |
| Fixed final validation NLL, step 1080 | 2.531569853 |
| Training phase wall time | 284.998 s |
| Sampled whole-GPU memory peak | 7,266 MiB, 0.2 s sampling |
| Baseline step-120 raw C compiled | 0 / 160 |
| Fixed final step-1080 raw C compiled | 0 / 160 |
| All-case model matches in either evaluated arm | 0 / 160 |
| Model targets reaching execution | 0 |
| Original reference cases independently rechecked | 10,240 |
| Test model examples / external training rows | 0 / 0 |

The intermediate step-240 checkpoint is retained by validation NLL only. Its
functional accuracy **has not been measured**; the complete 160-target C
comparison is between the predeclared baseline and fixed final checkpoint.
The final checkpoint is not promoted. A lower intermediate NLL is not evidence
of working decompilation, and neither evaluated arm produced compilable C.
No failed generation or compilation target was removed or repaired.

`checkpoint-selection.json` identifies the selected NLL checkpoint/hash and
ignored local weight location. Both selected and final weights were downloaded
and SHA-verified; the parent checkpoint's SHA remained unchanged. The actual LR
transition and parent SHA/step are recorded in `training-report.json`.
`source.json` and `executed-driver.py` match by SHA; exact source/payload hashes,
GPU/CUDA/Rust versions and binary identity are recorded. Intermediate states
remain as metadata; only final and in-job NLL-best weight tensors are exported.
Historical best checkpoints from other runs require their own retained artifact.

The sampler resumes at epoch 0/cursor 120 and ends at epoch 2/cursor 120.
The metrics contain every step 121..1080 and only training IDs. Epoch 1 visits
all 480 training IDs once. Because the run starts inside an epoch and each
new epoch shuffles independently, do not assume every ID occurs exactly twice
in these 960 updates. All validation checks use the full 160-example split.

`verification.json` records source/archive/checkpoint integrity, sample coverage
and **54 passing checks**: 22 Rust, 8 corpus Python, 11 evaluation Python,
8 controller tests, 1 real CPU LR-resume integration test and 4 intake tests.
Rust formatting and Clippy with warnings denied passed. Model quality remains
separate from software-test success.

`github-license.json` records GitHub detecting Apache-2.0. Newly generated owned
corpora use Apache-2.0; historical frozen corpus license metadata and lock hashes
are preserved for these checkpoint-compatible experiments.

All owned Colab sessions were released after verified artifact download.
`sessions-after.txt` and `usage-after.txt` record zero active assignments and
0.00/hr usage, with account balance 242.18 CUs. The observed account balance
change includes setup/retries and other ledger timing; it is not a GPU price.

The next substantive prerequisite is an expanded, independently validated
source/IR corpus. The present corpus contains only 53,908 supervised completion
byte tokens per epoch and does not justify proceeding to 20B training.
[20B scaling objective and gates](../../docs/scaling.md).
