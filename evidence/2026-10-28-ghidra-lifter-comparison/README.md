# Ghidra 12.2 vs Fission raw P-code on the owned corpus

First step of the Kuna / Ghidra / Fission three-track plan (all SLEIGH lineage):
can the same functions be lifted by Ghidra, and does the model-facing input change?

## Setup

* Ghidra **12.2 PUBLIC** (`ghidra_12.2_PUBLIC_20261016.zip`, 569,712,708 bytes), official
  GitHub release `Ghidra_12.2_build`, SHA-256 `2c2cd776db3c1f6a097ba418922378ac3649cadd5e844a6f5f088ae628a7be67`
  (matches the digest GitHub publishes). Installed in a new folder outside the
  repository; the existing `~/ghidra` source tree (12.3 DEV) was not used.
* Ghidra 12.2 requires **JDK 25+** (`application.java.min=25`); the Homebrew `openjdk`
  25.0.1 already on the machine was used by setting `JAVA_HOME` for the command only.
* `scripts/ghidra/DumpPcode.java` runs under `analyzeHeadless -noanalysis`: imports
  each of the 48 corpus ELFs (x86:LE:64:default), disassembles from every `corpus_*`
  symbol and writes raw (not high) P-code up to the first terminal flow. 1,762
  functions, a few seconds per ELF. No analyzer, decompiler or label is used.
* `scripts/ghidra/compare_lifters.py` normalizes operands (unique temporaries renumbered
  by first use, LOAD/STORE space-id operand masked) and compares to the Fission raw dumps.
* `scripts/ghidra/compare_prompts.py` converts the Ghidra dump into Forge's raw schema
  (straight-line functions) and compiles it with the corpus lanes using the *Fission
  profile*, so only the P-code differs.

## Results (1,762 functions)

| Check | Result |
| --- | --- |
| Raw P-code identical (ops, operands, addresses) | **1501 (85.2%)** |
| Raw P-code different | 261, all in `shl`/`shr`/`mul` families |
| First difference of those 261 | `INT_NOTEQUAL` vs `INT_EQUAL` (260), `COPY` vs `INT_SLESS` (1) |
| Compiled prompt identical, compact lane | **1762 / 1762** |
| Compiled prompt identical, compact-dec lane | **1762 / 1762** |
| Compiled prompt identical, return-u32 SSA lane | **1762 / 1762** |

The differing functions share the data flow into EAX (the first six operations of a
sample `a >> 5` are identical); they differ only in how flag updates guarded by the shift
count are expanded (Ghidra uses `INT_MULT`/`COPY` selects, Fission `BOOL_NEGATE` plus
`INT_AND`/`INT_OR`). Those flags are dead for a low-EAX return, so the output-driven slice
removes them and the prompt is unchanged.

The compiled `x86-64.sla` differs: Ghidra 12.2 `de384b09...1e19`, Fission's pinned
`d5adc314...5432481` (the Forge profile). The difference is therefore a SLEIGH
specification version difference, not a lifter bug.

## Limits

Straight-line, pure, low-EAX functions of a one/two-operation grammar, one ISA, clang -O1,
no branches, calls or memory data. The prompt equality relies on dead flags being sliced
away and says nothing yet about control flow, calls or Kuna. The converter reuses the
Fission profile (`fission_commit`/`sla_sha256`), so a lifter-agnostic profile is still
needed. A first converter bug (signed vs unsigned constants) was fixed before the numbers
above; the 1501 raw-identical count did not depend on it. Kuna was not run.
