# Corpus training, long inputs, and validation — 2026-10-20 KST

This records the first bounded training pass over the owned synthetic corpus.
It demonstrates training/validation/resumption, **not functional decompilation
accuracy, a real-project result, or a DecBench score**. Everything ran on local
CPU/F32. No Colab runtime or Hugging Face publication was used.

## Full training pass

The model in [`tiny-corpus-8k.json`](../../configs/model/tiny-corpus-8k.json) has
70,512 parameters, hidden width eight, one attention head/layer, and an
8,192-byte-token context. Fixed settings: seed 42, AdamW LR 0.001, batch size one,
query tile 256, 480 updates, full validation every 120 steps, checkpoints every
120 steps. No setting was selected against test results.

All 480 released training pairs were visited **exactly once**. Their IDs exactly
match the frozen train split; every validation check used exactly the 160 frozen
validation IDs. The test split had zero model examples. The separate corpus audit
checked all split artifacts, including test integrity, before training.

| Global step | Full validation examples | Completion-token NLL | Perplexity |
| --- | ---: | ---: | ---: |
| 0 | 160 | 5.559394 | 259.6654 |
| 120 | 160 | 3.960521 | 52.4846 |
| 240 | 160 | 3.052101 | 21.1598 |
| 360 | 160 | 2.504900 | 12.2423 |
| 480 | 160 | 2.176332 | 8.8139 |

The best post-update validation checkpoint was step 480. Validation aggregates
19,444 scored source/EOS tokens, excluding prompts and padding. These are
teacher-forced next-token predictions on shared C syntax and held-out generator
families. They do not test free-running C generation, compilation, or execution.
Loss improvement must not be described as functional recovery accuracy.

The run processed a maximum training sequence of **4,878** shifted byte tokens.
All 800 release rows fit the configured 8,192 context; this run does not claim
that an exact 8,192-token input was executed, nor were the longer test rows used
for training. Training plus periodic validation/checkpoint writes took
36.781 seconds, excluding setup and the initial validation sweep.
`/usr/bin/time -l` reported maximum resident memory of 821,362,688 bytes
(approximately 783.3 MiB) for the actual CLI process. This is a CPU process RSS
measurement, not GPU allocator memory or an 8K/100M capacity estimate.

[Full report](epoch-report.json), [all train/validation/checkpoint events](epoch-metrics.jsonl),
and [coverage/provenance verification](epoch-verification.json) preserve counts,
IDs, losses, model/config, binary hash, sampler state, and checkpoint digests.
Large weights and the complete corpus remain in ignored local `outputs/`.

## Resumption, isolation, and long-input checks

The [reproducible acceptance driver](../../scripts/corpus/train_probe.py) ran:

- Two updates, then one resumed update, versus three uninterrupted updates.
  Safetensors model weights **and both Adam moments** were byte-identical;
  sampled IDs, sampler position, and final validation loss matched exactly.
- Changed validation settings, corrupted weights, and a corrupted sampler cursor
  were rejected before creating training output directories. Corrupted weights
  were also refused by schema-4 generation.
- The longest training row was selected by sequence length only (seed 619),
  trained once, and validated against all 160 validation examples. This separate
  run peaked at 626,425,856 bytes (approximately 597.4 MiB);
  [probe verification](probe-verification.json) records the actual CLI measurement.
- A four-byte generation request successfully loaded a schema-4 checkpoint.
  [Raw generation report](generation-report.json) checks compatibility only;
  there is no assertion that this tiny output is valid or useful C.

[Longest-input/full-validation report](longest-validation-report.json) records
that one-update experiment separately from the 480-update model. The first three
update resumption probe used a declared two-example validation prefix; it must
not be confused with full validation or the epoch experiment.

The step-480 checkpoint was separately resumed for one update with the same
full-validation schedule. [Resume report](epoch-resume-report.json) records step
481, epoch one/cursor one, and [verification](epoch-verification.json) confirms
its next training ID is the expected new-epoch permutation entry. This exercises
an actual saved epoch-boundary checkpoint, in addition to sampler unit tests.

Rust tests compare tiled versus full attention logits and parameter gradients,
check detached validation has no variable gradients, check token-weighted NLL
aggregation and unchanged weights, and test epoch completeness/boundary resume.
The gate passed 21 Rust tests, eight corpus Python tests including existing
release-integrity regressions, two evaluation Python tests, rustfmt, Python
compilation, and Clippy with warnings denied. Old smoke/checkpoint/generation
regressions still pass.

## Reproduction and limits

Use [the corpus-training commands](../../docs/corpus-training.md), including the
recorded full-epoch settings. Checkpoints use schema 4; generation and the
external-kit adapter accept this lane. The adapter was not used for a real
benchmark run here. Long-context GPU training, 100M capacity, model-quality
generation checks, and final test scoring remain unmeasured.

Query tiling retains exact causal context but still has quadratic training
activation growth. The trainer caches encoded data on the host and uses fixed-LR
F32 AdamW. Out-of-core loading, mixed precision, gradient clipping/accumulation,
and scalable attention kernels are not implemented. Exact resumption is verified
on this CPU/backend, not as a cross-hardware bitwise guarantee.

Execution preceded the publishing commit. Records intentionally identify base
`aa96c9e` plus a dirty worktree and the actual executable SHA; that base alone
does not contain this implementation. Best checkpoint paths in reports refer to
local artifacts, not downloadable model releases.
