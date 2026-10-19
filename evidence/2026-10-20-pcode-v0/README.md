# P-code+ v0 extraction and training — 2026-10-20 KST

This record establishes a bounded input/training path, not decompiler quality.
The fixtures are owned C compiled with Clang 22.1.8 for x86-64 ELF at `-O1` and
statically lifted by pinned Fission `f714ffdf656fbc25993a4e6ffb038bb851ac6ba5`.
No fixture ELF was executed, and no semantic-equivalence proof is claimed.

## Extraction

Five functions were attempted. Addition, unsigned division, and signed division
passed v0 validation. The branch fixture was rejected for multiple blocks; the
call fixture was rejected for an unsupported direct address space. Neither
failure became a training example. This selected set is not a general coverage
estimate. [capture.json](capture.json) records all attempts and diagnostics,
compiler options, binary/CLI/SLA-profile provenance, and original raw-file hashes.
Absolute local paths and verbose error command strings were shortened for
publication; the original report's SHA-256 is retained.

[The lifter profile](../../configs/lifter/fission-x86_64.json) was generated using
Fission's own compiled-SLA parser. The checked-in raw dumps, C inputs, labels, and
three training rows live under [fixtures/pcode](../../fixtures/pcode/).
Re-extraction with explicitly pinned resource directories reproduced those rows
byte for byte. Prompt snapshots show exactly the model's conditioning text:

- [Addition](forge_add.prompt.txt)
- [Unsigned division](forge_udiv.prompt.txt)
- [Signed division](forge_sdiv.prompt.txt)

These texts omit labels and instruction addresses, retain numeric-domain
opcodes and widths, and make storage/effect dependencies explicit.

## Training and resumption

On local CPU, the 61,888-parameter F32 model used three examples, padded sequence
length 951, a 260-token byte vocabulary, seed 42, and AdamW learning rate 0.003.

| Run | Updates | Initial / final loss | Report |
| --- | ---: | --- | --- |
| Initial | 8 | 5.560713 / 4.4019127 | [report](train-8-report.json) |
| Separate-process resume | 2 more | 4.4019127 / 4.1213517 | [report](resume-2-report.json) |
| Uninterrupted | 10 | 5.560713 / 4.1213517 | [report](train-10-report.json) |

The 8+2 and 10-update checkpoints were byte-identical, including AdamW moments.
Reload-logit and next-update errors were zero in all reports. A deliberately
changed effective-input hash was rejected before tensor loading or output
creation. [resume-comparison.json](resume-comparison.json) records the comparison
and shared checkpoint digest. Weights are ignored and not committed.

Loss on these three training fixtures is a software probe, not held-out accuracy.
CPU timings include the scope described in [training documentation](../../docs/training.md)
and are not CUDA throughput estimates. No Colab runtime was allocated for this
experiment. General CFG SSA, corpus training, generation, and DecBench scoring
remain pending; see [the executable contract](../../docs/pcode.md).
