# Model evaluation with DecBench

## Decision

Use [DecBench](https://github.com/Noelo-Lab/decbench) as the primary external
evaluation framework for Forge's dedicated neural decompiler. Start with the
published sample-set eval kit, which also provides a comparison surface for
existing coding-agent results. Larger corpus evaluation follows after a working
model and adapter exist.

This document defines the intended protocol. The fixed-batch training and input
adapter probes described elsewhere do not establish a model-quality result.
Bounded model generation and an external-kit adapter are implemented;
a real DecBench dataset scoring run and upstream submission remain pending. Reference revisions are recorded in
[`configs/evaluation/decbench.json`](../configs/evaluation/decbench.json);
null dataset/hash fields must be resolved before a scored run.

## Metrics and interpretation

Use upstream implementations and aggregation rules without local scoring edits.

| Report | Meaning | Interpretation |
| --- | --- | --- |
| GED | Distance between source and reconstructed control-flow graphs | Lower is better; zero is the upstream perfect structural match |
| Type match | Recovered variable types compared with DWARF ground truth | Report the upstream score and correspondence evidence |
| Byte match | Normalized assembly similarity after recompilation with the matching toolchain and flags | Retain upstream fixups; distinguish raw compilation from compilation after fixups |
| Union | Perfect on at least one measurable metric | Not an all-metrics pass and not semantic equivalence |
| Compilability and coverage | Output, parsing, and compilation outcomes with their denominators | Report failures, abstentions, unsupported targets, and missing toolchains separately |

Keep all three metric columns visible alongside Union. Report upstream values
and perfect rates with measured/eligible counts; do not substitute a custom
weighted score. An auxiliary all-three-perfect count, if added, must be labeled
separately and restricted to functions with all three metrics measurable.

These metrics do not execute reconstructed code to establish behavioral parity.
Any future execution-based evaluation is a separately named result on a vetted
corpus with its own harness and input provenance. Never interpret compilation,
GED zero, or byte-match similarity as proof of semantic equivalence.

## Target scope and comparisons

- Freeze the eval-kit archive and `functions.json` hashes before inference.
  Preserve the complete manifest and publish the attempted/supported subset.
- The first model targets C and x86-64. Select that subset from binary metadata
  before examining model output, and record every excluded target with a reason.
- Compare Forge, Fission, available LLM4Decompile results, and published AI rows
  only on identical targets, optimization levels, metric revisions, and
  denominator/failure-normalization policies. Missing comparable results remain
  unavailable; they are not zero scores.
- Existing Codex/Claude Code runs are tool-using agent systems. A single-pass
  model is a different inference protocol: record tools, retries, context,
  decoding settings, timeouts, and budgets rather than implying identical setups.
- Keep assembly-only and Fission-assisted model variants under separate run IDs.
  The assembly-only lane uses static non-decompiling inspection and no generated
  pseudocode. Assistance must be disclosed and cannot inherit the upstream
  coding-agent lane's tool-policy claims.
- A supported-subset result is not a full sample-set or full-corpus result.
  Published leaderboard snapshots are reference data, not results of a new run.

## Training separation

DecBench evaluation targets, ground-truth sources, published reconstructions,
and metric-derived repair feedback are excluded from training and checkpoint
selection. Use a separate development validation corpus for tuning.

Split training/development data by source project before creating compiler and
optimization variants. Audit shared projects, forks, duplicate source functions,
and binary variants against the evaluation manifest. Keep an overlap report;
do not claim contamination-free results solely from different filenames or
binary hashes. Record unknown pretraining overlap for third-party base models.

Freeze the checkpoint and decoding protocol before the external evaluation.
Any later target-informed training makes those targets development data and
requires a fresh holdout for generalization claims.

## Integration path

1. Fetch the official eval kit from the pinned dataset revision. Inspect its
   README and manifest; the kit contains malware-derived binaries and is for
   static analysis only. Do not execute or emulate kit binaries or reconstructed
   functions as part of this evaluation.
2. Generate reconstructed C for the requested function addresses. Preserve raw
   outputs, parse failures, timeouts, and explicit unsupported outcomes.
3. Use the kit's own schema: one combined `.c` file per attempted binary and
   `results/results.json` mapping emitted function names to requested addresses.
   Model inference may be per-function; packaging combines outputs per binary.
4. Include optional line-to-instruction mappings only when substantiated by the
   binary. Do not fabricate address evidence for type correspondence.
5. Validate and package with the kit's `python3 package.py`. Successful packaging
   validates submission structure; it does not compute the benchmark metrics.
6. For local scoring, install the pinned evaluator and its documented dependencies
   (including Rust Joern and required recompilation toolchains), then follow the
   upstream external-ingestion workflow. For source-based scoring, retain required
   preprocessed `.i`/`.ii` units; absent source CFGs must not be reported as GED zero.
7. Preserve `scoreboard.toml`, `function_results.json`, per-function diagnostics,
   and upstream finalization/overlay provenance. Record metric cache versions and
   missing-metric reasons alongside aggregates.

## Reproducibility record

Each run must identify the Forge commit, model/tokenizer repository and immutable
revision, checkpoint/config hashes, Fission revision if used, DecBench revision,
dataset revision, kit/target hashes, selection rules, compiler/tool versions,
metric cache versions, prompt/template, seed, decoding/retry settings, hardware,
wall time, and actual compute-unit use where available. Store exact per-target
outputs and errors. A model upload or successful training loss curve is not an
evaluation result.

## Upstream references

- [Metric definitions](https://github.com/Noelo-Lab/decbench/blob/ae070b885b4e005ed37510a2c3445af6284dc2b9/docs/metrics.md)
- [Backends and external submissions](https://github.com/Noelo-Lab/decbench/blob/ae070b885b4e005ed37510a2c3445af6284dc2b9/docs/decompilers.md)
- [Coding-agent input/output protocol](https://github.com/Noelo-Lab/decbench/blob/ae070b885b4e005ed37510a2c3445af6284dc2b9/decbench/decompilers/llm_dec.py)
- [Eval-kit format](https://github.com/Noelo-Lab/decbench/blob/ae070b885b4e005ed37510a2c3445af6284dc2b9/decbench/evalkit/templates.py)
- [Leaderboard](https://decbench.com/leaderboard/)

DecBench reference: Noelo Lab, *DecBench: A Benchmarking Suite for Evaluating
Perfect Decompilation* (2026), https://github.com/Noelo-Lab/decbench.

## Implemented external-kit adapter

`scripts/evaluation/decbench.py` now connects static Fission lifting, Rust SSA
conversion, Rust checkpoint generation, Clang syntax/identifier validation,
and the upstream external-results format. It does not run metric computation,
kit binaries, generated C, kit-provided scripts, or an upstream submission.

```sh
# Inspect support and input sizes without running the model.
python3 scripts/evaluation/decbench.py \
  --kit /path/to/frozen-kit --prepare-only \
  --representation pcode-plus-ssa-v1 --output outputs/decbench-inputs

# Run a schema-3 or schema-4 checkpoint whose training lane matches the requested IR.
python3 scripts/evaluation/decbench.py \
  --kit /path/to/frozen-kit --checkpoint /path/to/checkpoint \
  --representation pcode-plus-ssa-v1 --max-new-tokens 256 \
  --output outputs/decbench-run
```

Run from the Forge root with the pinned clean Fission release build, a matching
SLA profile, the Forge binary, and Clang on PATH. `--device cuda`/`metal` forward
the requested generation backend and require a matching Forge build; the
recorded adapter exercise used CPU. `--max-targets N` is an explicit budget:
targets beyond it remain in the report as excluded, not absent. Each subprocess
has a 120-second timeout. This initial adapter handles ELF64 little-endian x86-64
headers. It does not consult private source-language metadata and therefore must
not be called a C-only benchmark slice without an independently frozen selection.

`run.json` records every public target, file/manifest/profile/checkpoint hashes,
Forge binary/revision/dirty state, frontend revision, input sizes and CFG counts,
generation stop reasons, errors, and separate lift/generation denominators.
`target_*/` retains raw dumps, prompts, and exact model outputs. A separate
`results/` contains one combined C file per binary and `results.json`.

A function is emitted only after EOS, valid UTF-8, one top-level C function
identified by Clang, and syntax validity of the combined translation unit. Partial
or invalid outputs remain raw evidence, not repaired submissions. The initial
output contract excludes preprocessor directives; no reference-source headers,
fixups, inferred function names, or retries are injected. Syntactic validity
is a preflight gate, not a DecBench metric or proof of equivalence.

The adapter writes to a fresh directory and leaves the kit untouched. To package,
review the results and follow the frozen kit's trusted upstream packaging procedure;
packaging remains separate from metric scoring. The checked exercise used an
owned one-function kit with the pinned DecBench packager, **not the DecBench
dataset**. Its output was deliberately memorized during the connection probe,
so it is unsuitable for model-quality evaluation. The failure exercise retained
five prepared functions, one unsupported file, and one missing file; a budgeted
run retained two prepared and five excluded targets. See
[the evidence record](../evidence/2026-10-20-cfg-generation/README.md).

## Local functional validation

The [synthetic functional protocol](functional-evaluation.md) independently
measures full-validation generation, compilation, and bounded source-result
comparisons. It retains every failure and uses no test examples in the model.
Its uint32/Wasm32 scope and finite probes are distinct from DecBench metrics.
