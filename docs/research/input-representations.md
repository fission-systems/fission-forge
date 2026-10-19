# Input representation experiments

## Hypothesis

IR conditioning may let a small model devote more capacity to source recovery
and less to instruction decoding. This is untested. Shared IR opcodes do not
erase ABI, register-layout, endianness, or compiler differences automatically.

Compare the following lanes on the same function identities and source labels:

| Lane | Input | Status |
| --- | --- | --- |
| `asm` | Binary-derived assembly | Handwritten smoke fixtures implemented; real extraction pending |
| `pcode-raw` | Fission raw P-code with original semantic details | Typed import implemented as the substrate for v0; standalone raw-text training comparison pending |
| `pcode-plus-ssa-v0` | P-code contracts with scalar SSA and effect ordering | Stable straight-line subset |
| `pcode-plus-ssa-v1` | Reconstructed CFG, cell/effect phis, and opaque calls | Five owned fixture conversions and CPU training/generation connection checked; corpus evaluation pending |
| `vex` | VEX block statements, types, exits, next, and jumpkind | Proposed independent lifting comparison |
| `asm-pcode` | Both representations with an explicit boundary | Optional ablation after separate lanes work |

The first IR recommendation is P-code because Fission is already a Rust/SLEIGH
frontend. Source inspection of the pinned Fission checkout shows raw P-code JSON
with decode count, stop reason, template provenance, and a PcodeFunction. The CLI
was built locally and exercised on three small compiled x86-64 functions
(addition, unsigned division, and signed division). All three dumps reached
terminal control flow; division dumps retained `IntDiv`/`IntRem` and
`IntSDiv`/`IntSRem` respectively. This checks extraction and opcode distinction,
not semantic parity or general lifting coverage. Forge now has a strict straight-line dataset adapter and reproducible owned-C fixture capture; see [the v0 contract](../pcode.md).
VEX is a useful later comparison, with Python orchestration for PyVEX extraction
if needed; the learned model and trainer can remain Rust.

## P-code+ and SSA boundary

The working recommendation is `SLEIGH -> raw P-code -> P-code+ -> SSA input -> C`.
Start with a small x86-64 CPU subset. P-code+ supplies semantic contracts; SSA
supplies explicit definition/use relationships and merge points. HIR need not be
a mandatory dependency. The name P-code+ describes a proposed Forge contract,
not an existing upstream Ghidra format. A bounded version 0 is now implemented;
the wider contract remains a design proposal.

The Notion research notes distinguish three concerns:

- **Execution and compatibility:** retain P-code operations and varnode details
  as the baseline substrate, with the raw dump available for traceability.
- **Contracts:** preserve numeric domains, memory/effect ordering, and other
  facts that constrain legal transformations. Required meaning must not live
  solely in an optional comment that consumers can silently discard.
- **Advisory metadata:** probable types, names, confidence, and presentation
  hints remain distinguishable from verified semantic facts.

Unknown helpers and incomplete lifting cannot be repaired by adding metadata.
Record unsupported cases explicitly, and classify any lowering as lossless,
lossy with declared losses, or unsupported. Do not infer register-space IDs from
numeric literals: bind them to the pinned lifter's actual space definitions.
Version 0 handles overlapping register writes and conservatively orders memory
accesses. Version 1 adds resolved CFGs and loop merges, with calls represented as conservative
opaque effects. Precise ABI/callee analysis and upstream lifting parity remain
unproven; see [the v1 contract](../cfg.md).

The broader FIR/FSL migration is outside the initial model's prerequisites.
These Notion documents are exploratory design sources, not implementation
contracts already satisfied by Fission:

- [P-code, P-code+, FIR compatibility strategy](https://app.notion.com/p/3c2744e97b6f8144b498d4bf6edbe2f6)
- [FIR semantic preservation and IR boundaries](https://app.notion.com/p/3c2744e97b6f81c8a46cc07f50bd05e7)
- [Migration and experiment plan](https://app.notion.com/p/3c2744e97b6f81ea8ddfe39b6b4cfc79)

Public readers may need workspace access for those notes. The constraints in this
document stand on their own; no Notion access is needed to build the repository.

## Serialization constraints

- Preserve operand widths, signed/unsigned operations, integer/float distinctions,
  address spaces, memory endianness, aliasing-relevant offsets, and effect order.
- Preserve block identities, conditional/unconditional edges, calls, returns,
  indirect targets, and unresolved exits. Instruction-local P-code branches must
  not be confused with machine-address branches.
- Rename temporary identifiers deterministically. Retain register identity and
  overlapping-register ranges; ABI-role names require actual calling-convention
  evidence. Do not relabel every register as a source argument.
- Branch addresses can become consistent labels while retaining the mapping as
  provenance. Arithmetic constants, field offsets, and relocation identities
  are semantic data and cannot be blindly replaced with placeholders.
- Unsupported instructions, unknown userops/helpers, and partial lifting remain
  explicit failures or unsupported examples. Dropping them is not normalization.
- Treat a lifting library's text dump as an inspection format, not a stable
  training schema. Pin lifter and serializer revisions and test deterministic
  serialization and semantic-preservation properties.
- Do not feed source-derived types/names, DWARF ground truth, or reference C into
  model inputs. Any binary-derived symbol/context assistance must be recorded.

VEX has typed temporaries and explicit register/memory effects, but also helper
calls and lazy condition-flag state that require careful representation.
Raw P-code varnodes have spaces/offsets/sizes; high-level P-code after analysis
can add operations and recovered relationships. Keep raw, SSA/high P-code, and
Fission-normalized IR under different lane identities.

## Fair experiment

Use the same project-disjoint splits, function addresses, output targets,
model size, and evaluation protocol. Measure the input length distribution,
lifter success/coverage, truncation exclusions, training tokens, GPU time, and
DecBench metrics. Report both the matched-function intersection and each lane's
full attempted coverage. Compare equal-token and equal-compute budgets where
practical; identical step counts alone are insufficient when IR expands input.

First use a sequential serialization with explicit block edges. A graph encoder,
cross-architecture pretraining, ASM/IR alignment objective, and IR-to-C curriculum
are later hypotheses, not necessary dependencies for the first ablation.

An IR-conditioned run is a learned decompiler with a lifting frontend. Its
quality depends on both components, so record frontend failures and versions.
An IR frontend also differs from DecBench's existing assembly-only agent
protocol: compare scores on matched targets while disclosing the different
input/tool conditions.

## Primary references

- [Ghidra P-code reference](https://ghidra.re/ghidra_docs/languages/html/pcoderef.html)
- [angr VEX IR guide](https://docs.angr.io/en/latest/advanced-topics/ir.html)
- [Pinned Fission raw P-code CLI](https://github.com/fission-systems/Fission/blob/f714ffdf656fbc25993a4e6ffb038bb851ac6ba5/crates/fission-cli/src/cli/oneshot/raw_pcode.rs)
