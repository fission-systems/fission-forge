# Rust/Candle training

`train-smoke` is a bounded fixed-batch acceptance tool. For long-input epoch
sampling, separate validation, and corpus resumption, use
[`train-corpus`](corpus-training.md). The remainder of this page describes the
original smoke path.
It supports at most 32 JSONL pairs and refuses truncation or output overwrites.
Legacy assembly rows have `id`, `assembly`, and `source`; those pairs are
hand-authored. The versioned P-code+ v0 lane validates raw dumps, builds SSA
prompts, and reads compiled fixture pairs. See [IR commands and schema](pcode.md).
VEX remains a proposal.

## CPU

```sh
cargo test --locked --workspace
cargo run --locked --release -p fission-forge -- train-smoke \
  --config configs/model/tiny.json \
  --dataset fixtures/smoke/pairs.jsonl \
  --output outputs/tiny-cpu --steps 32
```

`training.safetensors` contains named weights and both AdamW moments;
`state.json` contains configuration, step, seed, tokenizer identity, and dataset
hash. Schema 2+ also records the effective batch hash, guarding against serializer
changes; reports identify all input representations. A report is written only after checkpoint prediction and next-step
equivalence checks pass. Checkpoint-reload checks use F32 absolute tolerance
`1e-5`. Loss reduction is reported rather than used as a general quality gate.

Resume in a separate process with the same config, data, seed, and learning rate:

```sh
cargo run --locked --release -p fission-forge -- train-smoke \
  --config configs/model/tiny.json \
  --dataset fixtures/smoke/pairs.jsonl \
  --resume outputs/tiny-cpu \
  --output outputs/tiny-cpu-resumed --steps 8
```

The fixed batch, absence of dropout, and deterministic initialization eliminate
sampler/RNG state from this probe. The separate corpus trainer records versioned epoch/cursor state
and effective split hashes to make resumption reproducible. No LR scheduler, gradient accumulation,
gradient clipping, or mixed precision exists yet. A separate bounded
[generation path](generation.md) loads model weights without optimizer state.
Schema 3 saves also record training input representations and reject mismatched
lanes during generation.

## Colab A100

The Rust/CUDA build uses `--features cuda` and the trainer uses `--device cuda`.
`scripts/colab/transformer_probe.py` orchestrates a source-only upload, installs
Rust if necessary, builds the CUDA executable, then runs tiny/approximately-100M
probes. It does not implement model operations in Python or upload credentials.

Reproduction with the Colab CLI:

1. Package only `Cargo.toml`, `Cargo.lock`, `crates/`, `configs/model/`, and
   `fixtures/smoke/` as `forge-source.tar.gz` (paths relative to the workspace).
2. Allocate a named A100 session using `colab new -s <name> --gpu A100`.
3. Upload the archive to `/content/forge-source.tar.gz` and the driver to
   `/content/transformer_probe.py` using `colab upload -s <name>`.
4. Launch the Python orchestration driver through `colab exec`. For long builds,
   spawn it as a subprocess with output redirected to a log, then inspect
   `/content/forge-probe-status.json` and `/content/forge-transformer-probe.log`.
5. Download `/content/forge-probe-results.tar.gz` on success; on failure download
   the log/status and any partial reports. Always stop the named runtime after
   collecting results with `colab stop -s <name>`.

Throughput is measured for the exact fixed batch and F32 configuration, including
the final loss forward pass and scalar synchronization but excluding model setup,
compilation, checkpoint validation, and serialization. It is not steady-state
long-context corpus throughput. Memory is sampled every 0.5 seconds across the
whole GPU and may miss short-lived peaks; it is not an allocator peak metric.

Both smoke configurations use a 260-token byte vocabulary. The larger one is
14 layers, hidden size 768, 12 heads, and a 3072-wide feed-forward network.
It exists to measure feasibility; two updates are not pretraining.

## Corpus preparation

The [synthetic bootstrap corpus](corpus.md) has frozen family-disjoint splits
and an independent Rust integrity audit. The separate
[corpus trainer](corpus-training.md) uses only training rows for updates, with
detached validation and test exclusion. Its 8K CPU acceptance configuration fits
this release; long-context GPU memory/throughput remain unmeasured.
