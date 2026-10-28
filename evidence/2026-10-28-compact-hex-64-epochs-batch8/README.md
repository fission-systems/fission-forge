# Control: compact lane with hex literals, 64 epochs, batch 8

Question: was the 64-epoch success of the L3 run (decimal literals, digit-isolated tokens) due to
those representation changes, or to training length? Same rows (1367 / 205 / 112), model, seed, LR 0.0003
and update count (10,936) as the L3 run; only the prompt/tokenizer differ: hex literals, normal BPE
(vocabulary 424-class learned on the train split, the compact-lane codec). Colab L4, 782 s; session
terminated; CUs not recorded. Corpus rebuilt on 2026-10-28 after an accidental `outputs/` deletion
(release files and tokenizer byte-identical; plan bound to the rebuilt lock digest).

## Training-row reproduction (456 rows, CPU generation)

| Checkpoint | Skeleton right | Whole text identical | Literals right | Literals above 15 right |
| --- | ---: | ---: | ---: | ---: |
| 8 epochs, batch 1 (earlier, final) | 418 | 8 | 9 | 0/352 |
| 64 epochs, batch 8, step 2560 (selected) | 439 | 22 | 22 | 13/352 |
| 64 epochs, batch 8, step 10936 (final) | **456** | **423** | **423** | **325/352** |

## Validation (205 held-out functions)

| Checkpoint | Compiled | Functions matched | Operators right | Constants right | Identical after a<->b swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 epochs, batch 1 (earlier, selected) | 203 | 3 (coincidence) | 174 | 15 | not computed |
| **64 epochs, batch 8, selected (step 2560)** | 205 | **1** | 179 | 12 | 10 |
| 64 epochs, batch 8, final (secondary, CPU generation) | not scored | 13 outputs identical to the label | 183 | 165 | 107 |

## Reading

* The model copies literals (training rows 423/456, held-out constants 165/205 for the final
  checkpoint) with hex literals and ordinary multi-digit BPE tokens, so the decimal rendering and the
  digit isolation tried in L3 are **not required**; the training length is what mattered. Hex-to-decimal
  conversion of 16-255 is learned too (325/352).
* The NLL-selected checkpoint is early (step 2560, before copying is established) and scores 1/205; the
  final checkpoint is far better on every copy metric. The primary result remains the NLL-selected arm.
* Compared with the L3 final checkpoint (constants 172/205, a<->b swap exact 172/205, 0 identical to the label),
  the hex lane has slightly fewer correct constants (165) and a mixed argument binding: 13 outputs get the
  letters right and 107 are right after the swap, so its binding is less uniformly reversed.

## Limits

Single seed; checkpoint selected by validation NLL reached only an early step; the final-checkpoint rows
come from CPU generation, not the frozen GPU protocol, and were not functionally scored with it; only validation
was used. No checkpoint is promoted.
