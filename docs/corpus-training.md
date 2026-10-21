# Long-input corpus training and validation

`forge train-corpus` trains on the **train split of an audited frozen release**.
It shuffles examples by epoch, saves model/AdamW/sampler state, and runs separate
teacher-forced validation. This is distinct from the older fixed-batch
`train-smoke` command. Test examples are never passed through the model by this
command; the corpus integrity audit still reads their artifacts and fingerprints.

## Run a bounded experiment

First [build and freeze a corpus](corpus.md). From the Forge root:

```sh
cargo build --locked --release -p fission-forge

target/release/forge train-corpus \
  --config configs/model/tiny-corpus-8k.json \
  --corpus outputs/corpus-bootstrap-v1 \
  --output outputs/corpus-training \
  --steps 480 --learning-rate 0.001 --seed 42 \
  --batch-size 1 --attention-chunk 256 \
  --validation-limit 0 --evaluate-every 120 --checkpoint-every 120
```

The 70,512-parameter CPU acceptance configuration has a context of 8,192 byte
tokens. With 480 training pairs and batch size one, 480 updates visit every pair
once. Its tiny hidden dimension is intended to test the pipeline, not establish
useful decompilation accuracy. Choose a fresh output directory. No runtime is
allocated and no Hugging Face/GitHub artifact is uploaded by the command.

Inputs are read incrementally as JSONL and normalized/tokenized on the host;
encoded train and validation rows are **cached in host memory**. This is not an
out-of-core streaming trainer. Only the current training batch or single
validation example is transferred to the device. Batches pad to their own longest
sequence. All train/validation examples must fit the configured context before
training starts; overlong rows are rejected, never silently dropped or truncated.
The retained 1..32 bound is per optimizer-update batch, not a corpus-size limit.
Batch size one is the measured long-input setting.

## Attention and validation memory

The decoder computes exact causal attention in query tiles. Each tile attends
to keys up to its end, masking future positions within the tile. It avoids a
full-length square mask and limits a single score operation to a tile, while
retaining every earlier position. There is no sliding-window approximation.
Tiled/full logits and parameter gradients are compared in the Rust tests.

**Training activations still grow quadratically with sequence length.** Autodiff
retains tile computations for backward; this is not FlashAttention, gradient
checkpointing, or a guarantee that a 100M model fits long contexts. Previous
A100 results used 57-token sequences and cannot establish 8K capacity.

Validation uses a detached weight snapshot, preventing variable/gradient-graph
retention. It processes one example at a time. Its NLL is the sum of per-example
loss times scored completion tokens, divided by the total scored tokens. It
includes source bytes and EOS, excluding prompt and padding, and reports
`perplexity = exp(NLL)`. A dedicated regression checks aggregation and unchanged
weights; detached inference also has no parameter gradients.

`--validation-limit 0` evaluates all validation examples. A positive value selects
a **fixed file-order prefix**, reports every selected ID and the full denominator,
and labels the result as a prefix probe. It is not a representative sample or a
full-split score. Changing the limit during resume is refused.

Validation runs initially, at global step multiples of `--evaluate-every`, and at
each invocation's final step. Checkpoints are written at validation steps and
`--checkpoint-every` multiples. The best post-update checkpoint is selected only
by validation NLL; the initial unsaved baseline is not a candidate. Segment-end
validation can add selection candidates when splitting an experiment into more
invocations. Exact training resumption concerns weights, optimizer, and sampler;
validation timing/selection history also depends on these explicit boundaries.

## Checkpoints and resumption

Each complete checkpoint lives under `checkpoints/step-XXXXXXXX/`:

- `training.safetensors`: model weights and both AdamW moment tensors.
- `state.json`: schema 4, model/optimizer settings, step/seed/tokenizer, corpus lock
  and effective encoded split hashes, epoch/cursor, schedule settings, best
  validation metadata, and a SHA-256 of the weights/optimizer file.

Snapshots are written to a fresh partial directory and renamed only after both
files are complete. Resume validates hashes, input lane, corpus identity, model,
learning rate, batch size, schedule, tokenizer, encoded train/validation content,
and sampler position before allocating the model. Partial directories are not
usable checkpoints. The sampler uses a versioned deterministic Fisher–Yates
shuffle per seed/epoch; stored epoch/cursor reconstruct the exact next example.
An update advances the saved sampler only after the optimizer update succeeds.

Resume for 120 **additional** steps into a fresh output directory:

```sh
target/release/forge train-corpus \
  --config configs/model/tiny-corpus-8k.json \
  --corpus outputs/corpus-bootstrap-v1 \
  --resume outputs/corpus-training/checkpoints/step-00000480 \
  --output outputs/corpus-training-resumed \
  --steps 120 --learning-rate 0.001 --seed 42 \
  --batch-size 1 --attention-chunk 256 \
  --validation-limit 0 --evaluate-every 120 --checkpoint-every 120
```

Generation accepts schema 4 and uses its recorded query-tile size; it verifies
the weights-file digest and input lane. The older `train-smoke` command resumes
its own schema 1/2/3 format, not corpus schema 4. Existing old checkpoints remain
supported for their original paths.

The latest checkpoint and historical best weights can be different artifacts.
Keep/transfer the best checkpoint separately if needed. On relocating a latest
checkpoint, a best-at-current-step path is rebased; a missing older best path is
reported as null while its score/hash remain recorded. This does not prevent
training resumption and cannot recover missing historical weights.

## Outputs and acceptance evidence

`metrics.jsonl` is flushed after each train/validation/checkpoint event. Train
events record sampled IDs, sequence length, completion tokens, and sampler state.
`report.json` records corpus/config hashes, backend, initial/final validation,
best checkpoint metadata, steps, counts, maximum training sequence, and final
checkpoint path. Throughput includes periodic validation and checkpoint writes,
but excludes setup and initial validation. It is not steady-state GPU throughput.
A failed command may retain flushed logs and earlier complete checkpoints; only
a successful run writes its final report.

A reproducible bounded integration check is available:

```sh
python3 scripts/corpus/train_probe.py \
  --corpus outputs/corpus-bootstrap-v1 --output outputs/corpus-training-probe
```

It checks two updates plus one resumed update against three uninterrupted
updates, including byte-identical weights/Adam moments, sample order, and final
validation. It rejects changed validation settings and corrupted checkpoint
weights, then deliberately selects the longest **training** example by sequence
length (not by model score) and evaluates the full validation split. A four-byte
schema-4 generation probe checks compatibility without asserting valid C.
On macOS, `/usr/bin/time -l` records peak resident memory per actual CLI process.
See [recorded execution](../evidence/2026-10-20-corpus-training/README.md).

The current optimizer is F32 AdamW at a fixed learning rate. Gradient clipping,
accumulation, mixed precision, distributed training, efficient attention kernels,
and large-corpus host-memory scaling remain future work. Teacher-forced loss
improvement is not functional C correctness or a DecBench score. Keep test data
untouched until the checkpoint and external evaluation protocol are fixed.

## Colab GPU corpus training

[The bounded GPU runner](colab-corpus.md) exercises the same corpus trainer on
a Colab A100, using the 5.38M 8K configuration. It now records a complete
480-update epoch, full validation, CUDA resume equivalence and GPU generation.
See [measured evidence](../evidence/2026-10-20-colab-corpus/README.md).
External Hugging Face datasets remain [staged research candidates](hf-datasets.md).

## Corpus-specific learned codec

`tokenizer-train --corpus ROOT --vocab-size 772 --output NEW_DIR` reads only
the audited training split to learn the reversible lexical BPE codec. The output
contains `tokenizer.json` and training-only row lengths. `tokenizer-stats` reports
lengths without allocating a model. The requested size is a cap: training stops
when no pair appears twice. `train-corpus --tokenizer PATH` requires model
vocabulary size `260 + merges.len()`, pins the train split, and embeds the codec
in schema-5 checkpoints. Resume requires identical codec, model, sampler and
validation settings. Omitting the option retains byte codec and schema 4.

The recorded [composition trial](composition-corpus.md) produces vocabulary 698.
No validation/test text participates in vocabulary learning. Token-weighted NLL
is comparable across checkpoints with the same codec; its units differ between
byte and BPE runs. Use raw C compilation/probe results for functional comparison.
