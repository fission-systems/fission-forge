# CFG, generation, and external-kit connection — 2026-10-20 KST

This is an owned-fixture integration record. It is **not a DecBench dataset run,
held-out model-quality measurement, ISA-wide equivalence proof, or model release**.
No Colab runtime was allocated. All training and generation here used local CPU.

## CFG and call representation

[Capture provenance](capture.json) records five owned C functions compiled for
x86-64 ELF with Clang 22.1.8 at `-O1` and statically lifted by pinned Fission.
All five pass v1 conversion: callee, condition-move diamond, integer loop,
direct call, and indirect call. The original raw dumps and five JSONL extraction
rows are in [fixtures/pcode-cfg](../../fixtures/pcode-cfg/).
Instruction-address branch targets are reconstructed even when the exporter
leaves an interior branch in one raw block. Prompt snapshots:

- [Diamond](forge_diamond.prompt.txt): three reconstructed blocks from one raw block
- [Loop](forge_loop.prompt.txt): five blocks, including a self-loop and backedge phis
- [Direct call](forge_direct_call.prompt.txt): opaque call with returning-continuation unknowns
- [Indirect call](forge_indirect_call.prompt.txt): unknown target value and the same conservative effects

The Rust suite compares raw P-code execution, generated SSA execution, and
independent wrapping-u32 oracles on 42 diamond and 64 loop inputs. Both
representations agree on all 106 cases. Opaque calls are structurally checked for
fresh register definitions, including SP; the test interpreters refuse to execute
them. Target resolution, partial-lift rejection, undefined temporary rejection,
and instruction-scope handling are also checked.

This deliberately selected fixture set is not representative coverage. Scalar
and effect phis do not establish alias-aware memory analysis or interprocedural
precision. Calls may change all observed register cells and RAM, trap, or not
return; no ABI is inferred. See [the v1 contract](../../docs/cfg.md).

## Training and generation

Exactly one small callee example was deliberately memorized by the 61,888-parameter
F32 model: 160 updates, seed 42, AdamW learning rate 0.005, prompt 446 bytes,
shifted sequence length 499. [Training report](training-report.json) records loss
5.5494685 → 0.00716038 and zero checkpoint-logit/next-update errors.

[Generation report](generation-report.json) records exact bytes/tokens, model,
checkpoint/state/prompt digests, CPU backend, and EOS termination. The model
reproduced the same 51-byte training C label. [Generated fixture C](generated-owned-fixture.c)
contains that text. This tests model inference plumbing and memorization only;
it gives no estimate of unseen-function accuracy. The long loop prompt was
converted and tested, but was not trained with the 1,024-token model.

The checkpoint was built before the publishing commit. Reports therefore retain
the observed base revision, dirty-worktree indication, and exact Forge binary
hash. Do not read the base revision as the source of these uncommitted features.
Weights and kit binaries remain ignored; they are not committed.

## External-kit adapter and packaging

A one-function **owned** kit was created solely to exercise the DecBench schema.
The model had trained on that exact function. The adapter emitted C after EOS,
UTF-8, Clang function identification, and combined-translation-unit syntax gates.
[Run record](adapter-run.json) and [results mapping](results.json) retain the
actual target and output. No private source metadata was used for the prompt.

The standalone packager was read from DecBench commit
`ae070b885b4e005ed37510a2c3445af6284dc2b9`, rather than executing arbitrary
kit-supplied code. It accepted one C file / one owned target.
[Packaging record](upstream-packaging.json) contains its source hash, exit status,
stdout, and ZIP digest. This validates format, not GED, types, byte match, Union,
or official leaderboard acceptance. No upload or upstream submission occurred.

Failure/coverage exercises use a separate owned kit with seven public targets:

| Record | Prepared | Unsupported format | Missing-file failure | Budget exclusions |
| --- | ---: | ---: | ---: | ---: |
| [All targets](adapter-prepare.json) | 5 | 1 | 1 | 0 |
| [Budget of two](adapter-budget.json) | 2 | 0 | 0 | 5 |

All seven targets remain in each report; denominators are not reduced to the
successful subset. The all-target fixture binary was only statically inspected.

## Reproduce and limits

See [CFG capture](../../docs/cfg.md), [training/generation](../../docs/generation.md),
and [the external-kit adapter](../../docs/evaluation.md). Focused checks:

```sh
cargo fmt --all -- --check
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
python3 -m unittest discover -s scripts/evaluation -p 'test_*.py'
```

The recorded gate is 16 Rust tests plus two Python tests. Source/kit/checkpoint
freezing, project-disjoint corpus training, long-context GPU measurement,
held-out DecBench target selection, and actual metric computation remain separate
work. Packaging or a memorized fixture must not be reported as benchmark quality.
