# Fine-tuning a pretrained 0.5B code model on the same rows and prompt

Question: is the from-scratch 3.4M model the limiting factor? Same data, prompt and
scorer; only the model changes.

## Setup

* Model: `Qwen/Qwen2.5-Coder-0.5B` base, revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301`,
  Apache-2.0 (card tag and `LICENSE` file), 494,032,768 parameters, public/not gated.
* Data: the compact-lane `shallow-wide-v3` rows: **1367 train / 205 validation**
  (lock `f6a04c77...`), prompts rendered with `forge pcode-compile` by
  `scripts/finetune/prepare_data.py`. The 112 reserved test rows are never loaded.
  Prompt = lane text + `### C\n`; label = the C source + EOS.
* Full fine-tune, fp32 weights with bf16 autocast, AdamW (betas 0.9/0.95, no weight
  decay), LR 2e-5 with 20-step warmup and cosine decay to 10%, batch 8, 3 epochs
  (513 steps), seed 42, clip 1.0, completion-only loss. **One configuration, chosen
  before seeing any validation result; nothing was tuned.**
* Checkpoint selection: lowest token-weighted validation NLL, evaluated every 25
  steps, before any functional scoring (same rule as the Forge runs). Selected =
  **step 50** (NLL 0.1703; NLL 2.056 before training, 0.296 at the end).
  The final step-513 checkpoint is reported as a secondary arm; choosing it because
  it scores higher would be selection on the functional score, so the primary result
  is the NLL-selected arm. NLL is per Qwen token and not comparable to Forge's.
* Greedy decoding, 128 new tokens, raw text scored unchanged by
  `scripts/evaluation/score_external.py` (same compiler, Wasm runtime, probes and
  `functional.assess` as `score_gpu.py`; it reproduced the earlier scratch result
  exactly: 203 compiled / 3 matched / 1047 cases). No output exceeded the 512-byte cap.
* Hardware: Colab L4, training 183 s, torch 2.11.0+cu130, transformers 5.20.2.
  The session terminated. Account CUs were not recorded. Weights were not saved.

## Results (205 held-out validation functions, 13,120 cases)

| Model | Compiled | Functions matched | Case matches | Operators+args (constants masked) | Constants right | Distinct outputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch 3.4M, compact lane | 203 | 3 (coincidences) | 1047 | 42 | 15 | 23 |
| **Qwen 0.5B, NLL-selected (step 50)** | **205** | **79 (38.5%)** | 6328 | 84 | 80 | 170 |
| Qwen 0.5B, final (step 513), secondary | 205 | 97 (47.3%) | 7019 | 98 | 101 | 165 |

Breakdown (matched / rows):

| | One-operation (b op k), 107 rows | Two-operation, 98 rows |
| --- | ---: | ---: |
| Selected, step 50 | 3 | 76 |
| Final, step 513 | **0** | **97** |

Two-operation functions with held-out argument roles and 14 different literals
per family are copied and composed correctly (every family 14/14 except one 13/14
in the final arm). The model reads the operator, argument register and literal
from the prompt. This is the capability the scratch model never showed.

The one-operation failure is systematic, not noise: in the final arm **101 of 107**
one-operation outputs end with `<< 3u)` or `>> 3u)`, and 56 contain the correct
expression plus that extra wrapper (`((b + 2u) << 3u)` for `(b + 2u)`). In the
training split every `b`-first function is a two-operation template (the role
holdout puts all one-operation training functions on `a`), so the model learned
"b first implies a second operation". This is a data-design confound, not a
failure to read the input.

## Training-row reproduction (selected checkpoint, 456 training rows)

Exact text 171/456 (scratch: 8/456); literals right 178/456 (scratch 9/456);
constants above 15 right 106/351 (scratch 0/352); operators+arguments 394/456
(scratch 418/456). Step 50 is only about 0.3 epochs, so training rows are not yet
memorized; the scratch model had seen them eight times.

## Conclusion and limits

On identical rows, prompt and scorer, a pretrained 0.5B model fine-tuned for 183 s
solves the literal-copy and role-composition problems that the scratch 3.4M model
could not after ~12 minutes of training: 79/205 (NLL-selected) vs 3/205. The
remaining failures are explained by the training distribution. Caveats: single seed
and one configuration; 145x more parameters plus pretraining, so this is not an
architecture-controlled comparison; the task is a synthetic one/two-operation
grammar, not real code; only validation was scored (test split and the original
nested 130-function task untouched; no DecBench score); the final-checkpoint
score is secondary. No model is promoted or released.
