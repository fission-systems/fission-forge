# Colab A100 training probes — 2026-10-20 KST

These are recorded reports from the Rust/Candle F32 fixed-batch trainer, not
benchmark scores or pretrained model releases. The run began on October 19 UTC
(October 20 in Korea). Both probes used the four handwritten assembly/C pairs in
`fixtures/smoke/pairs.jsonl`, with padded sequence length 57.

| Files | Meaning |
| --- | --- |
| `tiny-cuda-report.json` | 148,608 parameters, 32 updates, checkpoint and resumed-update checks |
| `100m-cuda-report.json` | 99,927,552 parameters, two updates, same checks |
| `*-memory.json` | Whole-device memory sampled every 0.5 seconds; includes baseline and may miss peaks |
| `environment.json` | Recorded GPU, compiler, CUDA, and Candle environment |
| `source.json` | SHA-256 of the source-only archive uploaded for this run |

The archive digest identifies the pre-commit probe upload; it is not a Git commit
ID or a claim that every final documentation/test file was present in that
archive. Model and training code were unchanged when these results were added.
The reports also record the fixture content hash and full model configuration.
No checkpoint weights or credentials are included.

`training_seconds` measures updates plus final loss computation, excluding build,
setup, checkpoint validation, and serialization. The two-update 100M timing is
particularly unsuitable for extrapolating sustained training cost. Memory is an
observed sample maximum, not a guaranteed capacity bound for longer inputs.
Checkpoint equality covers this fixed batch, deterministic initialization, and
optimizer state; future data sampling, dropout, or scheduling need more state.

See [training commands and reproduction](../../docs/training.md). The runtime was
stopped after reports were collected.
