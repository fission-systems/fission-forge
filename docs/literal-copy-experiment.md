# Literal-copy experiments (proposed, not yet run)

Motivation: [constant-collapse evidence](../evidence/2026-10-28-constant-collapse/README.md).
The compact-lane model reproduces expression structure but not literals, even on
training rows (9/456 literal matches; 0/352 for constants above 15), and its
loss sits near the 0.101 nats/token "literal never copied" floor. All variants
below reuse the same **1367 / 205 / 112** rows, 3.37M RoPE model, seed 42,
LR 0.0003, 8 epochs, L4. Each learns a fresh train-only BPE for its own prompt.
Single seed; changes are made one axis at a time.

## Predeclared measurements

1. **Train-row literal match** with `constant_reproduction.py --split train --stride 3`
   (456 rows). Baseline 9/456. Copy is *learned* only if at least 90% match.
2. **Training NLL below the floor** (about 0.101 for this data): impossible
   without copying, so it is a loss-side confirmation.
3. **Validation literal match** and functional score with the unchanged scorer.
   Baseline 15/205 literal, 3/205 functional (the three coincide with literal 7).
4. Skeleton/operator counts, to confirm nothing else regresses (baseline 42/205
   and 174/205).

## Variants

| ID | Change | Isolates |
| --- | --- | --- |
| L1 | Render prompt literals as decimal: unsigned for logic/arith, **signed** decimal for two's-complement immediates (`#-2:8` rather than `#fffffffffffffffe:8`). New lane id, e.g. `pcode-ssa-return-u32-compact-dec-v1`; a pure function of the raw dump, no labels. | hex-to-decimal conversion and two's-complement decoding |
| L2 | Isolate digits: BPE never merges digit bytes (spec records the flag; train-only vocabulary otherwise unchanged), in prompt and label. | digit-aligned copying vs atomic multi-digit tokens |
| L3 | L1 and L2 together. | whether the two interact |
| L4 | Move the literal-bearing operation closer to the answer boundary (put the exit map first) or echo literals in a header line derived from the IR. | distance from literal to answer |

Run L3 first as the best-justified combination; ablate L1 and L2 only if L3
helps. A pure `a + k` template is excluded from L-variants because the audit
forbids sharing shapes across splits.

## Isolating learnability from the task (recommended before spending more L4 time)

**P0, single-template interpolation probe.** Train on `a op k` for 200 values of
`k` and hold out the other values of the *same* template. This separates "cannot
learn to copy" from "cannot compose roles". The corpus audit intentionally
rejects equal erased shapes across splits, so P0 needs an explicitly labelled
diagnostic mode (for example `scope=constant_interpolation_diagnostic`) that
skips only that cross-split shape rule, never produces headline numbers and is
kept in its own directory. If P0 fails with L3, the cause is optimization or
architecture (batch 1, step count, attention), not the representation; if P0
succeeds but the shallow task does not, the cause is composition/data.

**P1, optimization controls on the existing lane** (cheap, no new data):
batch 8 with the same epochs (`--batch` supports 1..32; fewer, less noisy
updates), and 2x epochs. Report train-row literal match and the 0.101 floor.

## Decision table

| Outcome | Reading | Next step |
| --- | --- | --- |
| L3 train copy at least 90%, validation literal high | Tokenization/representation was the barrier | Adopt L3, then retry role binding |
| L3 train copy at least 90%, validation literal low | Copy learned, role/context generalization missing | Role curriculum/auxiliary tasks |
| L3 and P1 both fail on train rows | Copy circuit not forming at this scale/steps | Auxiliary copy loss, longer schedules, larger model |
| P0 fails, P0 with larger model passes | Capacity-limited | Scale model before more data work |

## Implementation notes

* L1: rendering lives in `slice::compile_compact`; add a literal formatter over
  the existing `#hex:size` tokens and a size-aware sign rule. Unit-test with
  `forge_callee` and a subtraction fixture (negative immediate), and re-run
  `corpus-semantics`.
* L2: `tokenizer-train --isolate-digits`; the tokenizer spec and its SHA-256
  must record the flag so checkpoints cannot be decoded with the wrong codec.
* None of these touch validation/test row bytes; re-freeze references per lane.

## Outcome of L3 (run 2026-10-28)

L3 (decimal literals plus digit-isolated tokens) did not help: 0/205 functions,
152/205 compiled, literals right in 14/205, and the final checkpoint still copies
only 10/456 training literals. See
[evidence](../evidence/2026-10-28-decimal-digit-literals/README.md). Remaining
candidates: the P0 interpolation probe and the P1 optimization controls (batch
size, step count); number formatting is no longer the leading explanation.
