# Scaling objective and current training stage

The user-confirmed long-term target is **approximately 20 billion model
parameters**. This is a research objective, not an implemented
or currently trainable model size on the present Colab allocation.

## Resource floor under the current implementation

The implemented trainer uses dense F32 weights, F32 gradients and two F32 Adam
moment tensors. For P parameters, these four arrays alone require **16P bytes**.
For 20,000,000,000 parameters that is 320,000,000,000 bytes (about 298 GiB), before
activations, temporary optimizer tensors, KV state, checkpoint copies and CUDA
workspaces. F32 weights alone require 80 GB. The measured Colab A100 has 40 GiB
of VRAM. Dense from-scratch 20B training does not fit this runtime.

For an architectural arithmetic example, the existing untied-embedding decoder
with hidden width 6144, 44 layers, FF width 24576, 48 heads and 8192 learned
positions would have 19,985,952,768 parameters. This is **an analytical example,
not an allocated or validated training configuration**. Its formula is:

```
P = 2*vocab*hidden + positions*hidden
    + layers*(4*hidden^2 + 2*hidden*ff + 4*hidden) + 2*hidden
```

Adding a size to a JSON file does not implement a distributed trainer. Current
code has no mixed-precision training, activation checkpointing, FlashAttention,
optimizer/parameter sharding, distributed checkpointing or multi-host sampler.
CPU offloading is not a demonstrated practical workaround. Quantized adapter
fine-tuning is a different experiment from the selected from-scratch objective.

## Gates before scaling

| Stage | Exit evidence |
| --- | --- |
| Current 5.38M model | Resume the actual checkpoint, full held-out generation, raw C compilation and bounded execution comparison |
| Data scale | Real-source provenance and license checks, compile/lift/IO validation, project-disjoint splits and benchmark overlap exclusions |
| 100M then 300M experiments | Measured longest-input memory/throughput, reproducible checkpoint recovery, token budget and functional validation on the same frozen protocol |
| Larger single/multi-GPU models | Mixed precision and memory-saving kernels validated for both forward/backward; deterministic data and recovery contracts |
| Approximately 20B | Confirmed cluster allocation and budget, tested optimizer/parameter sharding, distributed recovery, sufficient audited tokens and independent functional evaluation |

Each architecture is a distinct model experiment. Changing layer counts or widths
cannot resume the existing checkpoint directly; any transfer, distillation or
weight expansion must be implemented and evaluated explicitly. Training tokens,
model parameters, wall time and functional accuracy are separate quantities.
A 20B-token budget has not been authorized or consumed by this run.

## Current bounded continuation

The first GPU pilot ended at step 480. The next predeclared experiment resumes
those same 5,380,608 parameters and Adam state for **1,920 additional updates**,
ending at step 2400 (five complete training passes in total). It retains the
same corpus lock, seed, optimizer LR 0.001, batch 1, query chunk 256, full
validation and 120-update evaluation/checkpoint cadence. It does not secretly
replace the optimizer state or change LR on resume.

Both the baseline and fixed final checkpoint generate all 160 validation inputs
on CUDA with a 256-token budget. The local evaluator checks unchanged C against
the existing compiler/Wasm contract and all 64 probes per target. Functional
scores are not used to choose an intermediate checkpoint. Validation NLL best
metadata is retained independently, and the test split remains unused.

Repeated exposure to 480 synthetic training pairs does not supply a real-project
corpus or establish readiness to scale. If this bounded experiment still fails
to generate correct C, the measured failure is recorded and becomes an input to
data/model changes; it is not solved by claiming that a larger model exists.

## Validation regression and explicit LR recovery

The step-2400 fixed-LR experiment regressed validation NLL (2.547 to 3.152),
and both arms compiled 0/160 outputs. The response is not unlimited repetition
at the same LR. A separately declared recovery experiment starts from the
previous validation-NLL best checkpoint at step 120, retains weights, Adam
moments, sampler and optimizer step, and explicitly changes LR from 0.001 to
0.0001 for 960 additional updates (final step 1080). Selection uses validation
NLL only, not functional score or the test split. This is a separate branch
from step 2400, not a continuous 3360-step checkpoint lineage.

The recovery is one bounded diagnostic experiment, not an open-ended validation
hyperparameter search. A small synthetic corpus cannot justify a 20B training
run regardless of repeated epoch count. The next substantive scaling prerequisite
is a larger, independently validated source/IR corpus with enough C grammar and
algorithm coverage. Family-disjoint synthetic validation is deliberately stricter
than random constant-variant splitting, but covers only two unseen generator
families and cannot substitute for project-level evaluation.

Measured training supervision is only **53,908 completion byte tokens per epoch**
(source UTF-8 bytes plus EOS), across 480 optimization pairs / 240 unique C
labels. Four additional passes consumed 215,632 completion tokens, not new data.
These byte tokens are not directly comparable to another model's BPE-token
training budget. They show why corpus scale, coverage and independent evaluation
must grow before treating a 20B parameter target as an executable training plan.
