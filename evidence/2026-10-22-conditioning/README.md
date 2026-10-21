# IR consistency and frozen-checkpoint conditioning — 2026-10-22 KST

**The owned IR/source consistency gate passes. Functional C generation remains
0/130 from the previous frozen generation experiment; no new generations or
training updates were performed in this diagnostic run.**

## Full-release IR audit

`forge corpus-semantics` verified the existing corpus artifacts and frozen
release, then compared both raw P-code and normalized SSA with the source
oracle for all **1221 rows: 903 train / 130 validation / 188 test**. Every row
ran 64 probes under two register fill states (0 and 0xa5), producing **156,288
comparisons per interpreter, zero mismatches**. Per-row counts and mismatch
lists are retained in `ir-semantics-report.json`.

The explicit x86 SysV ABI zero-extends u32 arguments into RDI/RSI, initializes
RSP to 0x1000, maps a return slot containing 0xfeedbaad and reads low EAX. Both
return targets and return values are checked. Unmapped RAM reads, opaque calls,
unsupported operations and fuel exhaustion reject execution. Scalar widths
are restricted to 1..16 bytes and register offsets to the bounded file. Raw
execution has a 20,000-operation limit; SSA has a 1000-block limit.

This promotes the existing fixture interpreter into a bounded corpus audit;
raw/SSA share arithmetic helpers, so they are not independent CPU emulators.
The independently computed Python oracle and previously checked native C form
the source reference. Finite probes on this pure-u32 grammar do not prove
whole-domain equivalence or support for general memory, calls or arbitrary x86.
Test labels were used only for IR consistency; test model examples remain zero.

## Fixed diagnostic protocol

The previously selected 3,507,712-parameter RoPE checkpoint at step 768 remains
unchanged. Its weights hash is
`399863fe1c05373083a537008f390d5fb21f182a552c934fa5fce82a81032b45`.
The corpus lock and 698-token train-only BPE match the previous experiment.

All 130 validation targets are scored with teacher forcing in two arms:

1. The target's own normalized IR prompt, followed by the complete original C
   and EOS.
2. The prompt from row `(i + 17) % 130`, followed by the **same target C and EOS**.
   Every donor is checked to belong to a different template family.

The rotation and 130-row contract were fixed in code before GPU execution.
There is no donor search, optimization, checkpoint selection or test inference.
The common header is `unsigned int f(unsigned int a, unsigned int b) { return `;
its token boundary is checked against each encoded target. Aggregates weight
tokens, not rows. `conditioning-report.json` preserves all row IDs, donor IDs,
prompt hashes, per-row sums and aggregate counts.

| Teacher-forced region | Tokens | Own input NLL | Wrong input NLL | Wrong minus own |
| --- | ---: | ---: | ---: | ---: |
| Complete C + EOS | 7946 | 0.531415 | 0.609941 | 0.078526 |
| Common header | 3120 | 0.009017 | 0.009017 | approximately zero |
| Body + EOS | 4826 | 0.869144 | 0.998437 | 0.129293 |

Own-input total loss is lower on **76/130 rows**. The recomputed correct-arm
NLL matches the previous selected-validation value within 1e-6. A worse score
under this wrong-input control demonstrates some conditioning, but does not
explain all generation failures or measure generated-C accuracy. Prompt length
and instruction mix also change in the control; it is not a clean intervention
on semantics alone. Teacher forcing supplies earlier correct target tokens,
which greedy generation does not receive. The near-zero common-header loss
also contributes substantially to the low full-completion NLL.

The previous **16/130 compiled / 0/130 all-case functional matches** therefore
remain the latest generation measurements. This diagnostic is not DecBench or
a final-test result. It does not justify a parameter-count increase.

## Execution and provenance

One named **L4 23,034 MiB** session was used. Rust/Candle built with CUDA;
`environment.json`, `binary.json`, `source.json` and `payload.json` bind hardware,
compiler, executable and source/artifact hashes. Diagnostic command wall time
was **23.99 seconds**, including corpus verification; sampled whole-device
memory peak **330 MiB** at 0.2-second intervals can miss brief peaks. Build took
3m34s. The phase limit was 300 seconds inside the existing bounded controller.
No optimizer or gradient update ran. The driver verified the input weights and
state before evaluation and unchanged weights afterward. Downloaded archive
and local checkpoint hashes were independently checked.

An observation timeout reused the same VM. The session was terminated, and
`colab usage` / `colab sessions` subsequently showed **238.65 CUs, 0/hour,
zero assignments and no server sessions**, versus 238.75 CUs before allocation.
The 0.10-CU difference is an account snapshot, not a universal GPU hourly price.

Rust 27 tests and Clippy pass; controller tests report 12 run with one optional
skip (11 passed); all 11 corpus tests pass with the real composition fixture.
The exact executed controller/driver and validation logs are included here.
`source-verification.json` checks uploaded Rust/Cargo sources against the working
files. The base revision was f37e05b with these changes uncommitted during upload;
the per-file hashes, rather than that base revision alone, identify execution.
Large checkpoint/source archives remain in ignored local `outputs/`.

## Next experiment boundary

Investigate a compact, label-independent SSA input derived from return-value
and control dependencies, while retaining the existing full SSA as evidence.
Before training it, require the same raw/SSA/source probe gate and recheck split
collisions on the transformed prompt. It must not import reference expressions,
remove unknown side effects or silently drop calls. Then train a fresh bounded
small model and score all 130 unchanged raw C outputs with the existing frozen
functional evaluator. No such representation change or follow-up training is
claimed by this record. Increasing epochs or scaling toward 20B remains gated
on nonconstant held-out generated C matching all 64 probes.
