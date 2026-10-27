# L3: decimal literals + digit-isolated tokens (not supported)

Hypothesis (from `docs/literal-copy-experiment.md`): the scratch model fails to copy
literals because the prompt prints them in hexadecimal while the label uses decimal,
and because BPE fuses digits into atomic tokens. Both were changed together.

## Change (everything else as the compact-lane run)

* Lane `pcode-ssa-return-u32-compact-dec-v1`: the compact lane with every `#hex:size`
  constant printed in decimal; a constant whose sign bit is set and whose
  two's-complement magnitude is at most 65535 prints negative (`b - 2` shows
  `#-2:8`, not `#fffffffffffffffe:8`). Lossless given the size suffix.
* Tokenizer flag `isolate_digits`: every ASCII digit is its own segment, so BPE can
  never merge digits (no learned token contains two digits; vocabulary 346, mean
  completion 41.8 tokens). Omitted from the serialized spec when false, so older
  codec hashes are unchanged.
* Same rows: **1367 / 205 / 112** with `id` and `source` identical to the compact
  corpus in every split; semantic audit 215,552 comparisons per interpreter, 0
  failures; references re-froze (205 targets, 13,120 cases). Same 3.37M RoPE model,
  seed 42, LR 0.0003, 8 epochs / 10,936 updates, L4 (training and job time in
  `fresh-resources.json`, 721 s total, session terminated). Rust tests: 33 pass.
  Account CUs were not recorded. Run used the working tree with uncommitted edits.

## Results

NLL selection chose **step 2048** (NLL 0.2578; tokenizer differs, so NLL is not
comparable to earlier runs). Validation (205 functions, 13,120 cases):

| Run | Compiled | Functions matched | Case matches | Ops right | Constants right | Distinct outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compact lane (hex, merged digits) | 203 | 3 (coincidences) | 1047 | 174 | 15 | 23 |
| **L3, NLL-selected (step 2048)** | **152** | **0** | 288 | 78 | 14 | 10 |

Outputs use literal 13 in 195/205 cases. Reproduction diagnostic on 456 training rows
(`scripts/evaluation/constant_reproduction.py`, CPU):

| Checkpoint | Skeleton right | Whole text right | Literals right |
| --- | ---: | ---: | ---: |
| Compact lane, final (earlier run) | 418 | 8 | 9 |
| L3 step 2048 (selected) | 132 | 2 | 8 |
| L3 step 10936 (final, diagnostic only) | 382 | 8 | 10 (2 of 352 above 15) |

On the 205 validation rows the final checkpoint gets literals right in 17 (the
constants it emits cluster on 14, 4, 1) and skeleton in 0.

## Conclusion

The hypothesis is **not supported**: neither decimal literals nor single-digit
tokens made the scratch model copy constants, even on rows it was trained on, and
the NLL-selected checkpoint is worse than the compact-lane run on every measure.
Together with the pretrained fine-tune (`2026-10-28-pretrained-finetune`), which
copies literals with an unmodified Qwen tokenizer, this points away from number
formatting and toward the copy mechanism itself (optimization at batch 1, scale or
missing pretraining) as the limiting factor. Single seed; the two changes were
bundled, so the individual effect of each is not separated; only validation was
scored.
