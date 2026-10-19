# P-code+ v0 input lane

For CFG/loop phis and conservative call handling, use the separate
[v1 lane](cfg.md). This page documents the stable straight-line representation.

Forge now accepts a bounded, executable `pcode-plus-ssa-v0` lane. It converts
Fission raw P-code to a deterministic text prompt, then feeds that prompt to the
same byte-level completion trainer used by the assembly baseline. The model and
adapter are Rust; Python only orchestrates fixture compilation and extraction.

## Supported contract

Version 0 supports byte-addressed, little-endian x86-64, one straight-line block,
and a final raw `Return`. A strict whitelist covers integer/boolean arithmetic,
comparisons, extensions, pieces, counts, RAM loads/stores, and return. Width and
arity checks run before serialization. Float operations, userops, calls, all
branches (including instruction-local branches), multiple blocks, unknown direct
spaces, partial lifts, and unrecognized template provenance are rejected.

This is a small input contract inspired by P-code+, not the full FIR proposal or
an upstream Ghidra format. The adapter retains original opcodes and operation
order; it does not optimize or raise them into HIR. No CPU execution semantics
are added by renaming an operation.

- Each scalar definition gets a unique SSA ID and byte width.
- Register state is tracked per byte. Partial writes replace exactly those bytes;
  subsequent overlapping reads use slices or explicit little-endian assembly.
  Missing register bytes become entry live-ins, never inferred C arguments.
- Temporary bytes must be defined before use within their machine instruction.
  The temporary map is cleared at instruction boundaries.
- Loads, stores, division/remainder operations that may trap, and return consume
  a conservative effect-order token. This is an ordering chain, not alias-aware
  memory SSA or an exception model.
- All operations, including flag writes and stack/return-address effects, remain.
  Absolute instruction addresses become ordinal instruction labels; temporary
  storage offsets become definition/use relationships. Register offsets,
  arithmetic constant bit patterns, widths, RAM identity, and order remain.
  Embedded address constants are retained; this is not general relocation
  normalization.

The explicit scope excludes phi construction, loops, call clobbers, unknown CFG
edges, floating-point contracts, atomic operations, and additional architectures.
A single-block SSA input is not general function-level SSA support.

## Inspect an input

```sh
cargo run --locked -p fission-forge -- pcode-compile \
  --raw fixtures/pcode/forge_add.raw.json \
  --profile configs/lifter/fission-x86_64.json \
  --output outputs/add-ssa
```

This writes `prompt.txt` and `report.json` to a new directory. Invalid inputs
fail before creating that output directory. The profile declares architecture,
space roles, the Fission revision, and the compiled SLA digest. Profiles supplied
by a caller are declarations; offline conversion cannot prove they match an
arbitrary raw dump. The capture script below binds the actual SLA and binary.

The pinned profile's role IDs (`const=0`, `unique=2`, `ram=3`, `register=4`) were
read through Fission's own compiled-SLA parser. They are not universal constants.
The profile's SLA hash and repository revision must be updated together when
changing frontends.

## Train on captured fixtures

Three pairs are committed in `fixtures/pcode/pairs.jsonl`. They come from owned C
functions compiled into an x86-64 ELF and lifted statically, rather than handwritten
assembly. They test software plumbing, not model quality or instruction parity.

```sh
cargo run --locked -p fission-forge -- train-smoke \
  --config configs/model/tiny-ir.json \
  --dataset fixtures/pcode/pairs.jsonl \
  --output outputs/pcode-smoke --steps 8
```

The IR configuration has one layer, hidden width 32, and a 1,024-byte-token context.
The original assembly model configurations are unchanged. The 100M configuration
still has a 512-token context and cannot fit the longest v0 fixture; extending
its context requires a separate configuration and new CUDA measurements. Examples exceeding
context are rejected, not truncated. Four legacy assembly pairs remain valid.

IR JSONL rows contain `schema_version`, `id`, `input`, `source`, and `provenance`.
`input` holds the representation, profile, and typed raw dump. The trainer rebuilds
and validates the prompt, instead of accepting arbitrary preformatted text as
verified SSA. Canonical raw and source-label hashes are checked. The canonical
raw hash uses Rust serde serialization of the typed dump; the capture report
separately records the original dump-file bytes' hash.

Only the compiled prompt enters the input token stream. IDs, source labels,
source hashes, binary hashes, original instruction addresses, mnemonic comments,
and source paths are excluded. Source C enters the completion side only.
Checkpoint schema 2 records a hash of the effective batch shape and input/target/mask arrays so
that a changed serializer cannot silently reuse an incompatible checkpoint.
Schema 1 assembly checkpoints remain readable with their original provenance
checks; schema 1 IR resume is refused.

## Re-extract from C

Prerequisites: a clean pinned Fission checkout with `fission_cli` and
`fission-sleigh` built in release mode; Clang with ELF cross-compilation support,
LLD, `llvm-nm`, and the Rust compiler used for that Fission build. Fission is not
vendored or added as a mandatory Cargo dependency.

```sh
# Optional comparison clone; ignored by Forge.
git clone --depth 1 https://github.com/fission-systems/Fission.git external/Fission
git -C external/Fission fetch origin f714ffdf656fbc25993a4e6ffb038bb851ac6ba5
git -C external/Fission checkout --detach f714ffdf656fbc25993a4e6ffb038bb851ac6ba5
cargo build --locked --release --manifest-path external/Fission/Cargo.toml -p fission-cli
cargo build --locked -p fission-forge

python3 scripts/pcode/capture_fixture.py \
  --fission-root external/Fission \
  --forge target/debug/forge \
  --rustc rustc \
  --output outputs/pcode-capture
```

The compiler named by `--rustc` must match the built Fission rlib. The helper uses
thin LTO to link that release artifact. For the recorded local run this compiler
was Homebrew Rust 1.96.1; Forge itself used Rust 1.97.0.

The script compiles `functions.c` for `x86_64-unknown-linux-gnu` at `-O1`, finds
fixture addresses from the ELF symbol table, forces the x86-64 SLEIGH language,
and disables Fission analysis-database hints. It records every attempt and its
unsupported reason in `capture.json`, alongside supported JSONL pairs and raw
dumps. The target ELF is never executed. Compiler failures fail the command;
per-function conversion failures remain explicit records. Failed attempts are
not silently recast as successful training examples.

The recorded five-function experiment accepted add/unsigned-divide/signed-divide
and rejected branch/call functions. This deliberately selected set is not a
representative coverage estimate. See [recorded evidence](../evidence/2026-10-20-pcode-v0/README.md).

## Validation and remaining work

Regression tests compare overlapping register reads with an independent byte
array across 101 writes and 3,636 reads. Other tests cover relocation/temporary
renaming invariance, domain/effect retention, label isolation, digest tampering,
undefined temporaries, malformed widths/constants, and unsupported control flow.
These tests establish the adapter's bounded behavior; they do not prove the
upstream lift equivalent to the source or machine code.

The separate [v1 lane](cfg.md) now supplies resolved CFGs, loop merges, and
opaque calls; [generation](generation.md) and an [external-kit adapter](evaluation.md)
are also implemented. A project-disjoint corpus, more precise call/memory
contracts, and actual DecBench scoring remain pending. No DecBench score or
semantic-equivalence result is claimed by this input-lane experiment.

Primary semantics reference: [Ghidra P-code operation reference](https://ghidra.re/ghidra_docs/languages/html/pcodedescription.html).
