# Kuna vs Fission vs Ghidra raw P-code on the owned corpus

Question: does Kuna export raw P-code, and does the model-facing input change with the
lifter? Third track of the Kuna / Ghidra / Fission plan.

## Does Kuna export raw P-code?

Not from its CLI. `kuna decompile --raw` prints the decompiler's **post-simplification**
operations (for example `EAX = EDI >> #0x5`), and `kuna disassemble --json` prints
instructions only; neither is raw SLEIGH output. Kuna's Rust SLEIGH engine (`kuna-sleigh`,
the crate its own `golden_lift` test drives) does lift raw P-code, so
`tools/kuna-pcode-dump` is a small separate Rust program (path dependencies on the local
checkout, built in its own target directory; the Kuna tree is only read) that lifts each
`corpus_*` symbol instruction by instruction up to its first RETURN. Output has the same
JSON shape as `scripts/ghidra/DumpPcode.java`, so the existing comparison scripts are reused.
The only Kuna build step needed was `cargo build --release -p kuna-console` in its existing
(gitignored) `target` for the CLI helper used while probing; the dumper does not depend on it.

* Kuna checkout: local fork at commit `b637463a`, Apache-2.0 `LICENSE` text.
* Spec: Kuna's own `x86-64.sla`, SHA-256 `ce8eca07997d8ec6ad02dc906d3d642a8ca62d050909915effad0eb3223bd450`.
  Context defaults set from the pspec/golden fixtures: addrsize 2, opsize 1, rexprefix 0, longMode 1, bit64 1.
* Three different compiled `x86-64.sla` hashes are in play: Kuna `ce8eca07...`, Ghidra 12.2 `de384b09...`,
  Fission (Forge profile) `d5adc314...`.

## Results (1,762 functions)

| Comparison | Raw P-code identical | Different |
| --- | ---: | ---: |
| Kuna vs Fission | **1761** (99.94%) | 1 (`a >> 1`, same 16 ops in a different order) |
| Kuna vs Ghidra 12.2 | 1501 | 261 (260 first differ `INT_EQUAL` vs `INT_NOTEQUAL`) |
| Fission vs Ghidra 12.2 | 1501 | 261 |

| Compiled prompt vs Fission, Kuna-derived | Identical |
| --- | ---: |
| compact lane | **1762 / 1762** |
| compact-dec lane | **1762 / 1762** |
| return-u32 SSA lane | **1762 / 1762** |

Kuna and Fission agree on the shift/mul flag expansion and Ghidra 12.2 is the outlier;
all three produce identical model inputs because those flags are dead for the low-EAX
return and the output-driven slice removes them.

## Limits

Straight-line pure functions of a one/two-operation grammar, one ISA, clang -O1; no
control flow, calls or memory data. The Kuna dumper lifts with default context and stops
at the first RETURN. The converter reuses the Fission profile (`fission_commit`,
`sla_sha256`), so a lifter-agnostic profile is still pending. Kuna's decompiled C is not
evaluated here.
