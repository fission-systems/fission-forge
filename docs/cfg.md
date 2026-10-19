# P-code+ v1: CFG, loops, and opaque calls

`pcode-plus-ssa-v1` is a separate representation from the stable straight-line
[v0 lane](pcode.md). It is implemented in Rust and accepted by the trainer and
generator. Version 0 prompt bytes remain unchanged.

## Reconstructing control flow

Fission's raw block list is not used as the definitive CFG. For example, the
captured x86 `cmov` fixture contains a `CBranch` inside one exported block.
Forge orders operations by their sequence numbers, checks instruction-address
order, and splits at branch targets, branch fallthroughs, calls, and returns.
Fallthrough is the next P-code operation, including inside an instruction.
Resolved RAM-address branch targets map to the first operation at that machine
address. This reconstructs both the condition-move diamond and normal loop
backedges without requiring HIR.

All operations must be reachable from the entry in the reconstructed graph.
Unknown targets, indirect branches, constant-space instruction-relative branches,
unreachable trailing operations, incomplete lifts, and unsupported opcodes fail
explicitly. The current bounds are 512 decoded instructions, 4,096 operations,
128 blocks, and 256 register cells. A return must exist in the bounded graph;
pure non-returning loops are outside this first version. Frontend correctness
is still an upstream dependency.

## Storage and phi construction

Register ranges are partitioned at every read/write boundary. Each resulting
cell can therefore be written completely while scalar outputs still preserve
original widths. Definitions retain byte slices when a wider value supplies a
narrower cell. Temporary cells are scoped to one machine instruction.

Blocks initially receive symbolic cell parameters. Required phi operands are
completed from predecessor exit states after all bodies have been converted.
A dependency closure retains parameters used by operation operands, observable
exit register state, or other phi operands. This supports forward edges,
backedges, self-loops, and merges without assuming an acyclic graph. Entry live-ins
are explicit; a loop returning to entry has both entry and backedge predecessors.
Scalar IDs may have gaps because unused provisional parameters are omitted.
Trivial phis are currently retained.

Temporary phis are allowed across blocks split inside the same instruction.
At the first operation of a machine instruction, temporaries are undefined.
An undefined temporary or a temporary crossing an instruction boundary rejects
the input. Register and effect phis are evaluated on incoming edges; phi
assignments are simultaneous on a backedge.

Memory effects use a conservative path/loop ordering chain. They do not implement
alias analysis or a byte-addressed memory-SSA optimizer. Numeric-domain opcodes,
widths, pure operations, memory accesses, and potential arithmetic traps remain.
There is no exception-atomicity or rollback model. Machine-instruction grouping
is retained in the raw provenance, rather than modeled as a full exception
contract in the prompt.

## Calls are conservative abstractions

Raw P-code `Call`/`CallInd` supplies a branch hint, not a complete logical function
signature or return convention. Version 1 splits at each call and records an
opaque call that may read/write all registers and RAM, trap, or never return.
Every observed register cell, including the stack pointer, gets a fresh unknown
value on a possible returning continuation. No caller/callee-saved set, argument
list, return register, or stack adjustment is guessed. Raw stack operations
around the call remain visible.

This prevents accidental use of pre-call values after an unknown callee, while
being less precise than ABI-aware call analysis. It does not execute or inline
callees, prove that a call returns, or recover a callee's algorithm from its
address. Unresolved external/tail jumps remain unsupported. The test interpreters
refuse to execute opaque calls.

## Run and reproduce

```sh
cargo run --locked -p fission-forge -- pcode-compile \
  --representation pcode-plus-ssa-v1 \
  --raw fixtures/pcode-cfg/forge_loop.raw.json \
  --profile configs/lifter/fission-x86_64.json \
  --output outputs/loop-ssa

python3 scripts/pcode/capture_fixture.py \
  --fixture-dir fixtures/pcode-cfg --entry forge_loop \
  --representation pcode-plus-ssa-v1 \
  --rustc rustc --output outputs/cfg-capture
```

The capture prerequisites, compiler matching, and source/SLA provenance rules
are described in [the v0 extraction guide](pcode.md). Five owned compiled
functions are checked in: a condition-move diamond, an integer loop, a small
callee, a direct call, and an indirect call. All five pass conversion.

The largest captured prompt is 4,581 bytes, exceeding the current tiny IR model's
1,024-token context. The checked-in five-row file is an extraction fixture set,
not a batch that fits that model. Do not truncate it. Only the small callee was
used for the recorded training/generation connection probe. Longer-context
training needs a separate configuration and new memory/performance measurements.

## Validation evidence

Two test interpreters use different state representations: a mutable raw P-code
machine and an evaluator of the generated SSA blocks/phis. Across 42 diamond
inputs and 64 loop inputs, both agree with independent wrapping-u32 source
oracles. This includes zero iterations, many backedges, unsigned boundaries,
and arithmetic wraparound. Calls are tested structurally for fresh unknown
register values, including SP, and explicit unknown memory/non-return effects.
Malformed targets, partial lifts, undefined temporaries, and unsupported branch
forms reject explicitly.

These are bounded owned-fixture checks, not an ISA-wide equivalence proof,
interprocedural parity, or a DecBench metric. See [recorded evidence](../evidence/2026-10-20-cfg-generation/README.md).

References: [Ghidra operation semantics](https://ghidra.re/ghidra_docs/languages/html/pcodedescription.html),
[MLIR blocks and block arguments](https://mlir.llvm.org/docs/LangRef/#blocks).
