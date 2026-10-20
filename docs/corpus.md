# Synthetic bootstrap corpus and independent splits

The first corpus is **owned synthetic C**, not a collection of real projects.
Its purpose is to exercise reproducible source → x86-64 ELF → raw P-code →
P-code+ SSA v1 extraction, source-label checks, and split isolation before
corpus-scale model training. It does not establish general decompilation quality.

## Frozen experiment definition

[`bootstrap-v1.json`](../configs/corpus/bootstrap-v1.json) fixes seed 42, 40
constant variants per generator family, three optimization levels, and split
assignments **before compilation**:

| Split | Families | Source programs | Planned compiler variants |
| --- | --- | ---: | ---: |
| Train | affine, bitmask, bounded_loop, carry, rotate_mix, select | 240 | 720 |
| Validation | remainder, xor_fold | 80 | 240 |
| Test | clamp, quotient_round | 80 | 240 |

There are ten algorithm templates and 400 source programs, not 1,200 independent
algorithms. All variants of a family stay together. Validation and test therefore
contain generator families absent from training; this is **family disjointness**,
not independent real-project sampling. Test data must stay out of training,
checkpoint selection, and prompt/example tuning. Validation can select settings;
report its use separately from the final untouched test evaluation. Changing a
split after seeing model results starts a new experiment version.

No benchmark source is read by the builder. Actual overlap against a pinned
DecBench dataset has **not been measured**. Freeze and compare the real external
evaluation corpus before making a non-overlap or benchmark-quality claim.

## Build locally

Use the pinned Fission checkout, SLA, and build described in
[the capture documentation](cfg.md). A current Clang/LLD and `llvm-nm` must be
available on PATH; the recorded run uses Clang 22.1.8. No GPU is needed.
Run from the Forge root, choosing a fresh output directory:

```sh
cargo build --locked -p fission-forge
python3 scripts/corpus/build.py --output outputs/corpus-bootstrap-v1

target/debug/forge corpus-audit --root outputs/corpus-bootstrap-v1
FORGE_CORPUS_FIXTURE=outputs/corpus-bootstrap-v1 \
  python3 -m unittest discover -s scripts/corpus -p 'test_*.py'
```

The builder refuses an existing output path. It checks the clean pinned Fission
revision and SLA hash; overrides the lifter's resource directories; compiles C
at `O0/O1/O2` with explicit flags; and lifts each symbol with `--max-bytes` equal
to its ELF symbol size. Vectorization, inlining, and loop unrolling are disabled.
This deliberately narrow compiler distribution must accompany coverage reports.
Target ELF files are statically inspected, never executed.

Each source program has 49 boundary input combinations and 15 deterministic
random probes. A host-native executable calls the owned generated C functions;
its results must match a separate Python wrapping-u32 oracle before extraction
continues. Divisors are nonzero, shifts are 1..31, loops are bounded, and unsigned
int is required to have 32 bits. **These checks validate C labels on the host;
they do not execute the extracted x86 P-code/SSA or prove lifter equivalence.**
The existing diamond/loop interpreter regressions remain a separate narrow gate.

Python performs data generation and tool orchestration. Rust owns P-code
validation, prompt reconstruction, corpus auditing, and release serialization.
Model operations and training remain Rust/Candle.

## Audit and release contract

The Rust `corpus-audit` command independently reconstructs each prompt, verifies
raw/source/binary/profile hashes, checks the executable ELF symbol bytes, and
checks assignments against the configuration and program manifest. Every planned
program/optimization combination must have exactly one attempt record; every
supported attempt must have exactly one pool row. Unsupported attempts retain
diagnostics and remain in coverage denominators.

Fingerprints cover exact source, Clang AST with local/function names removed,
AST shape with literals erased, the finite oracle-output vector, machine bytes,
and normalized SSA prompt. Shared fingerprints union **whole families**. A
component spanning different assigned splits is quarantined, not reassigned.
Rows with identical prompts but different source labels are also excluded.
Within one split, identical prompt/label pairs retain only the first sorted ID.
A finite behavioral fingerprint is a conservative collision detector, not a
semantic-equivalence proof. Quarantine and deduplication counts are reported.

`--finalize` writes a new `release/` only if all three filtered splits are
nonempty. Without it, the command recomputes and checks the frozen release;
artifact, split-byte, or lock differences cause failure. Finalization is one-shot
and refuses an existing release. Hashes detect changes relative to the frozen
local artifact; this is not a signed attestation or verification of arbitrary
third-party corpus claims.

| Artifact | Meaning |
| --- | --- |
| `config.json`, `profile.json`, `manifest.json` | Split recipe, lifter/compiler identity and artifact hashes |
| `sources/`, `binaries/` | Generated source, host check harness, and static target binaries |
| `programs.jsonl`, `probes.json`, `native-verification.json` | Canonical C labels, AST/oracle fingerprints, probe set and host comparison totals |
| `attempts.jsonl` | Complete attempted denominator with successes and unsupported diagnostics |
| `pool.jsonl` | All supported typed IR pairs plus provenance/split metadata |
| `attempts/<id>/` | Raw dump and reconstructed prompt diagnostics; not included in the release lock |
| `release/{train,validation,test}.jsonl` | Disjoint deduplicated typed pairs, compatible with the existing data reader |
| `release/audit.json`, `release/lock.json` | Exclusions, coverage, context eligibility, split hashes and manifest hash |

Source labels use a canonical function name `f`, independent of compiled symbol
names. Names, family IDs, split membership, ASTs, and expected probe outputs never
enter the SSA prompt. Canonical naming avoids asking the model to guess original
symbol names. A future multi-function evaluation integration must define its
symbol binding separately; the current external-kit adapter performs no such
renaming.

## Training boundary

This release is data preparation, **not a newly pretrained model**. The current
trainer still accepts at most 32 examples in one fixed batch. It has no corpus
sampler, held-out loss loop, or sampler-aware resume state. Never concatenate
validation/test into its input to fill a batch.

The audit reports shifted byte-token sequence lengths and eligibility at fixed
context budgets. Overlong pairs must not be truncated. The original 1,024-token
IR configuration fits none of the recorded release rows. Supporting all rows
requires a larger context; memory/throughput must be measured for that context
before allocating a large training run. The earlier approximately 100M A100
probe used length 57 and says nothing about this corpus's long-context capacity.

Next steps are a bounded corpus trainer with train-only sampling and resumable
sampler state, frozen validation loss and generation checks, and independent
project-split source acquisition. Keep the final test and DecBench targets
untouched until their evaluation protocol and checkpoint are fixed.
