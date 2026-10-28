# Lifter tracks: Fission, Ghidra, Kuna

All three are SLEIGH-based, so raw P-code is the shared input. Status:

| Track | State |
| --- | --- |
| Fission | Implemented; the corpus and every lane are built from it |
| Ghidra 12.2 | Headless dump and comparison implemented (`scripts/ghidra/`); 1762/1762 prompts identical to Fission on the owned corpus, see [evidence](../evidence/2026-10-28-ghidra-lifter-comparison/README.md) |
| Kuna | Raw P-code via the Rust SLEIGH engine (`tools/kuna-pcode-dump`; the CLI exposes only post-simplification ops and C). 1761/1762 raw functions identical to Fission, 1762/1762 prompts identical in all lanes, see [evidence](../evidence/2026-10-28-kuna-lifter-comparison/README.md) |

Open work: a lifter-agnostic `Profile` (today `fission_commit` is Fission-specific), control
flow and call cases (where the sliced prompt may differ), and the pseudo-C tracks for the
post-processing plugin. Ghidra 12.2 requires JDK 25 or newer.
