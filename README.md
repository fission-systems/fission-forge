<p align="center">
  <img src="assets/forge-banner.svg" alt="Fission Forge — neural decompilation, trained from scratch" width="1200" />
</p>

<p align="center">
  <a href="https://github.com/fission-systems">Fission Systems</a> ·
  <a href="docs/training.md">Training</a> ·
  <a href="docs/research/input-representations.md">Input representations</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="docs/corpus.md">Corpus</a> ·
  <a href="LICENSE">AGPL-3.0</a>
</p>

# Fission Forge

**A Rust research workspace for a neural decompiler trained from scratch.**

Forge explores how far a small, dedicated model can go from binary-derived
intermediate representations to C. The model, automatic differentiation, and
training loop use Rust and [Candle](https://github.com/huggingface/candle).
[Fission](https://github.com/fission-systems/Fission) supplies the intended
SLEIGH/P-code frontend; [DecBench](https://github.com/Noelo-Lab/decbench) is the
selected external evaluation protocol.

The initial scope is **C functions on x86-64**. Our weights start from random
initialization. [LLM4Decompile](docs/research/llm4decompile.md) informs the research,
but Forge is an independent implementation rather than a fine-tune of its models.

## Current state

This repository contains a working **training prototype**, with CPU and Colab
A100 checks. It does not yet ship a generally useful decompiler or a pretrained model release.

| Component | What exists today |
| --- | --- |
| Rust causal decoder | Multi-head causal attention, learned positions, pre-layer normalization, GELU feed-forward layers |
| Training | Byte tokens, completion-only loss, AdamW, deterministic epoch sampling and resumable corpus checkpoints |
| Long inputs / validation | 8K-context CPU configuration, tiled causal queries, detached token-weighted validation; test split excluded from model batches |
| Checkpoints | Safetensors weights and optimizer moments; reload and next-update equivalence checks |
| Model sizes | 61,888-parameter IR probe, 70,512-parameter 8K corpus probe, 148,608-parameter assembly smoke model, and 99,927,552-parameter experimental configuration |
| CUDA | 148K and approximately 100M assembly probes exercised on a Colab A100 40 GB; IR lane tested on CPU |
| Input data | Owned synthetic bootstrap corpus with frozen family-disjoint train/validation/test splits, plus small assembly and P-code fixtures |
| P-code frontend | Typed raw-dump validation, pinned SLA profile, reproducible owned-C extraction, and explicit unsupported records |
| P-code+ / SSA | Stable straight-line v0 plus v1 CFG reconstruction, loop phis, instruction-internal merges, and conservative opaque calls |
| Generation | Bounded greedy byte decoding, exact output preservation, EOS/context/budget reporting, checkpoint lane checks; CPU validated |
| DecBench | Static external-kit adapter and pinned packaging format exercised on an owned fixture; benchmark scores pending |

The fixed-batch probe accepts at most 32 examples. The new corpus trainer samples
from all training rows, supports batches of 1..32, caches tokenized rows on the
host, and rejects truncation. Out-of-core streaming, mixed precision, and
functional generalization evaluation remain ahead. Loss reduction alone is not
evidence of decompilation quality.

## Research direction

```text
Binary → Fission raw P-code → P-code+ / SSA → Forge checkpoint → C bytes
                              v0 / v1        corpus / probe    greedy
                                                               ↓
                                                   DecBench external-kit format
```

Version 1 reconstructs CFG edges from operations, including branches inside a
machine instruction. It adds register/temporary phis, loop backedges, effect
merges, and opaque direct/indirect calls. Unknown calls invalidate observed
register state and may change memory or fail to return; no ABI is guessed.
Unresolved branches, unsupported operations, and incomplete lifts reject explicitly.
HIR is not required.

Five owned compiled fixtures pass v1 conversion. Diamond/loop SSA execution
matches raw P-code and independent integer oracles on 106 inputs. A separate
one-example memorization probe generated its C label and passed the pinned
DecBench packaging format. **This is plumbing evidence, not held-out accuracy or
a DecBench score.** Larger-scale training and scored evaluation remain ahead.

The [bootstrap corpus pipeline](docs/corpus.md) checks owned C labels on the host,
records every compiler/lifter attempt, filters duplicate and cross-split
fingerprints, and freezes separate train/validation/test files. This is synthetic
family-disjoint data preparation; real-project independence, larger-scale training,
and functional held-out quality remain unmeasured.
[Corpus training and validation](docs/corpus-training.md) now add epoch sampling,
long-input attention, separate validation NLL, and schema-4 resume state. A
70,512-parameter CPU run completed one pass over all 480 training pairs and
validated separately on 160 pairs: NLL 5.559 → 2.176. This measures next-token
prediction, not functional C accuracy. [Execution record](evidence/2026-10-20-corpus-training/README.md).

Read [v0 input commands](docs/pcode.md), [CFG/call contracts](docs/cfg.md),
[generation](docs/generation.md), and [recorded evidence](evidence/2026-10-20-cfg-generation/README.md).

## Run locally

Install a current stable Rust toolchain, then run from the repository root:

```sh
git clone https://github.com/fission-systems/fission-forge.git
cd fission-forge
cargo test --locked --workspace

cargo run --locked --release -p fission-forge -- train-smoke \
  --config configs/model/tiny.json \
  --dataset fixtures/smoke/pairs.jsonl \
  --output outputs/tiny-cpu --steps 32
```

The command writes a JSON report, `state.json`, and `training.safetensors`.
Choose a fresh output path for each run; existing directories are refused.
To resume for eight additional steps:

```sh
cargo run --locked --release -p fission-forge -- train-smoke \
  --config configs/model/tiny.json \
  --dataset fixtures/smoke/pairs.jsonl \
  --resume outputs/tiny-cpu \
  --output outputs/tiny-cpu-resumed --steps 8
```

A separate synthetic tensor check is available through
`backend-probe --output outputs/backend-cpu`. CUDA builds use `--features cuda`
and `--device cuda`; Metal builds use `--features metal` and `--device metal`.
GPU requests fail explicitly if their backend is unavailable. Metal has not been
validated in this project.

## Colab evidence

On **2026-10-20 KST**, the Rust/Candle CUDA implementation ran on a Colab
**A100-SXM4 40 GB**, using F32 and four examples with a padded sequence length of
57 tokens:

| Configuration | Updates | Initial → final loss | Observed device memory peak |
| --- | ---: | ---: | ---: |
| 148,608 parameters | 32 | 5.5613 → 0.4161 | 546 MiB |
| 99,927,552 parameters | 2 | 5.5390 → 3.5934 | 4,322 MiB |

Both runs reported zero error for checkpoint logits and the compared resumed
AdamW update. Memory was sampled every 0.5 seconds across the whole GPU and may
miss transient peaks. The 100M run establishes a short training/checkpoint
feasibility check; it does not establish long-context capacity, sustained
throughput, or model quality.

[Recorded reports and provenance](evidence/2026-10-20-a100/README.md) are committed.
[Training documentation](docs/training.md) explains measurement scope and Colab
reproduction. Python in [the Colab driver](scripts/colab/transformer_probe.py)
handles orchestration; model computation stays in Rust. Runtime allocation is
manual and should be stopped after collecting results.

## Evaluation

Use a pinned DecBench evaluator, dataset revision, and target manifest. Report
GED, type match, byte match, and the upstream Union score on matching scopes,
alongside lifting coverage and input conditions. DecBench evaluation targets
must remain excluded from training.

The external-kit adapter records every target, unsupported input, timeout,
context/generation failure, and raw output. It uses only the public target mapping
and writes one C file per binary. Scoring dataset revision, evalkit digest, and
target-manifest digest remain unset until a real benchmark run is frozen.
No DecBench score is claimed here. See
[the protocol](docs/evaluation.md) and [reference configuration](configs/evaluation/decbench.json).

## Repository map

| Path | Purpose |
| --- | --- |
| [`crates/forge/`](crates/forge/) | Rust model, byte codec, loss, optimizer, checkpoint checks, and CLI |
| [`configs/model/`](configs/model/) | Tiny and approximately 100M model configurations |
| [`fixtures/smoke/`](fixtures/smoke/) | Hand-authored assembly fixtures |
| [`fixtures/pcode/`](fixtures/pcode/) | Stable v0 fixtures and negative cases |
| [`fixtures/pcode-cfg/`](fixtures/pcode-cfg/) | Owned diamond/loop/call functions, real dumps, and v1 extraction rows |
| [`scripts/colab/`](scripts/colab/) | CUDA build and bounded probe orchestration |
| [`scripts/corpus/`](scripts/corpus/) | Owned C builder, release integrity checks and training/resume acceptance driver |
| [`configs/corpus/`](configs/corpus/) | Versioned source-family split recipe |
| [`scripts/evaluation/`](scripts/evaluation/) | Static DecBench external-kit adapter and parser checks |
| [`evidence/`](evidence/) | Small, recorded measurement reports; no model weights |
| [`docs/research/`](docs/research/) | Paper notes and IR design hypotheses |
| [`assets/`](assets/) | Self-contained SVG banner and mark |

The optional Fission comparison clone belongs in `external/Fission/` and is
ignored. Build outputs, local runs, and checkpoints in `outputs/` are also ignored.
No GitHub, Colab, or Hugging Face credentials belong in this repository.

## Development

```sh
cargo fmt --all -- --check
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
python3 -m py_compile scripts/colab/transformer_probe.py scripts/pcode/capture_fixture.py
python3 -m unittest discover -s scripts/evaluation -p 'test_*.py'
python3 -m unittest discover -s scripts/corpus -p 'test_*.py'
```

Tests cover causal masking, shifted completion loss, prompt/padding exclusion,
AdamW agreement with Candle, overlapping register SSA, domain/effect retention,
input isolation, and fail-closed IR validation. Runtime probes separately check checkpoint
reload and resumption. Any new IR lane needs deterministic serialization,
semantic-preservation checks, and explicit unsupported-input reporting before
corpus training.

## License

[GNU Affero General Public License v3.0](LICENSE). Upstream tools, datasets, and
model artifacts retain their respective licenses.
