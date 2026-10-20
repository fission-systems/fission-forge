# Full validation: generated C compilation and bounded execution

This evaluates the owned synthetic **validation** split: 160 compiler variants
of 80 source programs from two held-out generator families. It is not a real
project benchmark, a DecBench score, or a proof of semantic equivalence.
The test split is never passed to the model or used for checkpoint selection.

## Protocol and frozen experiment

[`functional-validation-v1.json`](../configs/evaluation/functional-validation-v1.json)
predeclares the baseline at step 480 and one additional-training experiment:
resume the same model/Adam state for 1,440 train-only updates (three epochs), then
measure **the fixed final step 1,920**. No functional score is used to choose
among its intermediate checkpoints. Settings, corpus lock and baseline weights
hash are fixed before the comparison.

Both arms use the same executable, 256-token greedy EOS/byte decoding, CPU,
30-second generation wall deadline, compiler/linker hashes and flags, Wasmtime
version, and frozen prompts/probes/references. Raw completion bytes are written
unchanged as `candidate.c`: no Markdown stripping, renaming, includes, wrappers,
repair, or reference-text insertion. The model only receives the SSA prompt;
reference C and expected outputs are consumed separately by the evaluator.

The corpus has 64 deterministic probe inputs per target (49 boundary
combinations plus 15 random inputs). Reference C is compiled to freestanding
Wasm32 and executed on all inputs, then checked against the frozen source-label
oracle previously checked against host-native C. All 160 reference variants must
pass all 10,240 comparisons before a protocol manifest is finalized. The
160-variant count includes two optimization inputs for each of 80 source
programs; these are not 160 independent algorithms.

The freeze records prompt/source/reference-module/probe/oracle hashes, compiler
and linker identities, evaluator/worker source hashes, Rust executable hash,
experiment plan, and runtime version. Evaluating against a changed implementation
or toolchain is refused. Choose fresh output directories; do not rewrite an old
protocol to accommodate different weights or results.

## Compilation and execution contract

The current corpus contains portable uint32 scalar functions with two uint32
arguments. Clang compiles raw C using C11, `-O1`, `-nostdlib`, trapping undefined
behavior sanitizers, and `--export=f`. Compilation success means **compilation
and linking into the measured Wasm module**, including the required export.
The runtime then checks the ABI `f(i32,i32)->i32`; returned bits are compared as
uint32. Helpers/globals can compile, but no host imports may be instantiated.

This is a freestanding Wasm32 subset, not general native C compatibility.
Preprocessor/hash/digraph/trigraph markers, `_Pragma`, and inline assembly are
rejected as source-contract failures rather than repaired. This also prevents
source-provided include paths from reading arbitrary host files. Standard-library
calls requiring host imports fail compilation/linking or the import gate.
Wasm32 has different pointer/long widths from x86-64; this is suitable for the
current uint32 source corpus, not a claim of arbitrary x86 binary/runtime parity.

The [Wasmtime worker](../scripts/evaluation/wasm_worker.py) supplies **no WASI,
host functions or imports**. Every input uses a fresh Store/instance so state
cannot leak between probes. Limits are:

| Resource | Bound |
| --- | --- |
| Wasm fuel | 100,000 per input |
| Linear memory | 1 MiB, one memory |
| Wasm stack | 256 KiB |
| Tables | One table, at most 1,024 elements |
| Module file | At most 1 MiB |
| Worker CPU | 5 seconds, hard limit 6 seconds |
| Worker wall time | 10 seconds for the target's probe sweep |
| Compiler | 20 CPU seconds / 30 wall seconds |
| Generator | 25 CPU seconds / 30 wall seconds |

CPU limits apply per process; parent processes kill the whole subprocess group
on wall timeout. File-backed
logs have file-size limits and core dumps are disabled. Guest code cannot access
host files, sockets or processes through capabilities because none are supplied.
These are guest-memory/stack and process-time bounds; total native JIT/Python RSS
is not capped by `RLIMIT_AS` on macOS. The trusted host compiler is not itself a
WebAssembly sandbox. Undefined-behavior instrumentation is not a proof that all
possible C undefined behavior is detected.

See the primary [Wasmtime Python API documentation](https://bytecodealliance.github.io/wasmtime-py/)
for fuel and Store limits; the project pins its runtime in
[`requirements-functional.txt`](../scripts/evaluation/requirements-functional.txt).

## Full denominator and failure accounting

Every expected ID has one result, including generation failure/timeout, invalid
UTF-8, empty/contract-rejected source, compilation failure/timeout, ABI or import
rejection, execution timeout, fuel exhaustion, traps/runtime errors, mismatches,
and all-case matches. No failed row is removed from the denominator.

- Compile rate: compiled-and-linked targets / all 160 targets.
- All-case match rate: targets matching all 64 probes / all 160 targets.
- Case match rate: matched input cases / all 10,240 expected cases, including
  cases not executable because earlier stages failed.

Reports separately expose how many targets reached execution and how many input
cases actually executed. A zero match rate with zero compilable targets must not
be described as 10,240 model executions that produced wrong values. Generation
stop reasons are retained; even budget-terminated raw UTF-8 is passed unchanged
to the compiler, and malformed partial C naturally fails.

`run.json` starts in `running` state, per-target results and `records.jsonl` are
written as work completes, and a complete `report.json` is written only after the
full unique ID denominator is checked. Interrupted/invalid runs are not scores.
The comparison verifier checks the frozen protocol, record hashes, ID sequence,
raw-byte identity and result aggregation before exporting both arms.

## Reproduce

The baseline weights remain local experiment artifacts. Recreate the earlier
[480-update run](corpus-training.md) if needed. This recorded plan identifies the
actual baseline SHA; on another backend/host, floating-point or serialization
changes may yield another hash. Freeze a new explicitly versioned plan containing
that actual hash instead of silently changing the recorded experiment.

```sh
python3 -m venv outputs/eval-venv
outputs/eval-venv/bin/python -m pip install -r scripts/evaluation/requirements-functional.txt
cargo build --locked --release -p fission-forge

outputs/eval-venv/bin/python scripts/evaluation/functional.py freeze \
  --corpus outputs/corpus-bootstrap-v1 --output outputs/functional-frozen

outputs/eval-venv/bin/python scripts/evaluation/functional.py evaluate \
  --frozen outputs/functional-frozen \
  --checkpoint outputs/corpus-training-epoch-v1/checkpoints/step-00000480 \
  --output outputs/functional-baseline

target/release/forge train-corpus \
  --config configs/model/tiny-corpus-8k.json --corpus outputs/corpus-bootstrap-v1 \
  --resume outputs/corpus-training-epoch-v1/checkpoints/step-00000480 \
  --output outputs/functional-additional-training \
  --steps 1440 --learning-rate 0.001 --seed 42 --batch-size 1 \
  --attention-chunk 256 --validation-limit 0 --evaluate-every 120 --checkpoint-every 120

outputs/eval-venv/bin/python scripts/evaluation/functional.py evaluate \
  --frozen outputs/functional-frozen \
  --checkpoint outputs/functional-additional-training/checkpoints/step-00001920 \
  --output outputs/functional-after

outputs/eval-venv/bin/python scripts/evaluation/compare_functional.py \
  --frozen outputs/functional-frozen --baseline outputs/functional-baseline \
  --improved outputs/functional-after --training outputs/functional-additional-training \
  --corpus outputs/corpus-bootstrap-v1 --output outputs/functional-comparison

outputs/eval-venv/bin/python -m unittest discover -s scripts/evaluation -p 'test_*.py'
```

Keep the executable/evaluator/compiler fixed between freezing and both evaluations.
A new implementation requires a new frozen protocol and both arms rerun.
[Recorded results](../evidence/2026-10-20-functional-validation/README.md) include
all raw completions encoded losslessly, per-target failures, both checkpoints'
hashes, training coverage and the comparison.

## Replay the published C without checkpoints

The recorded evidence contains all 320 completions as lossless base64, the
frozen inputs/reference vectors, and the source-generator recipe. On the
recorded compiler/runtime, the following regenerates original C, rechecks all
10,240 reference cases, and recompiles/evaluates both sets of unchanged outputs:

```sh
outputs/eval-venv/bin/python scripts/evaluation/replay_functional.py \
  --evidence evidence/2026-10-20-functional-validation \
  --output outputs/functional-replay
```

This replays C compilation/execution metrics, not model inference. It requires
the recorded toolchain and evaluator hashes and refuses modified artifacts.
The complete generation experiment still requires its checkpoint or a recreated
training run. The published replay succeeded for every target in both arms.
