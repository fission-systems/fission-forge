# Compact return-u32 prompt: operators are read, constants still are not

Question: is the 800-byte return-u32 SSA prompt too noisy for a 3.4M model? Only
~12 of its bytes carry the operator, argument registers and literals; the rest
is constant ABI plumbing (live-ins, single-entry phis, effect chain, the
return-address load and `Return`).

Change (single axis): new lane `pcode-ssa-return-u32-compact-v1`
(`slice::compile_compact`). It keeps the SSA dataflow and low-EAX exit map but
resolves live-ins/phis to register names and drops the stack-return epilogue.
It accepts only one straight-line block whose sole memory effect is the
return-address load from the entry RSP; anything else is rejected, so it is not a
general lane. Example, `((b ^ 7u) - a)` (233 vs ~800 bytes):

```
pcode-ssa-return-u32-compact-v1 x86_64 le observe=low_EAX
%0:4=Copy(r30:4)
%1:8=IntZExt(%0)
%2:4=IntXor(%1[0:4],#7:4)
%3:8=IntZExt(%2)
%4:4=IntSub(%3[0:4],r38:4)
%5:8=IntZExt(%4)
exit_regs(r0=%5[0:4])
```

## Controls

Rows come from the same `shallow-wide-v3` corpus: **1367 / 205 / 112** with
`id` and `source` identical to the SSA-lane release in every split. Same 3.37M
RoPE model, seed 42, LR 0.0003, batch 1, 8 epochs / 10,936 updates, L4. Only the
prompt and the train-only BPE (632 tokens; was 692) differ. Max sequence 497
(was 810). The IR semantic audit passes 215,552 comparisons per interpreter with
no failures (finite probes, pure-u32 ABI). Validation references re-froze
identically (205 targets, 13,120 cases). Rust tests: 31 pass.

Run: training 269 s, job 716 s, session terminated. Account CUs were not
recorded. Selection chose the final step 10936 (NLL 0.2422; SSA lane 0.2573).
Execution used the working tree with uncommitted Rust edits at the time.

## Outcome (205 validation functions, 13,120 cases)

| Run | Compiled | All-case matches | Operators right | Operators+args right (constants masked) | First arg right | Constants right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SSA lane, LR 0.0003 | 188 | 1 | 87 | 13 | 42 | 15 |
| SSA lane, LR 0.003 | 86 | 0 | 56 | 0 | 16 | 7 |
| **Compact lane, LR 0.0003** | **203** | **3** | **174** | **42** | **100** | **15** |

(Counts are post-hoc multiset/skeleton comparisons of the returned expression.)

The three matches (`two_mul_b_0005`, `two_shl_b_0005`, `two_sub_b_0005`) are all
the variant whose literal is 7. The model emits literal **7 in 156/205**
outputs and 13 in 34, i.e. a near-fixed constant; the matches are coincidences
with it, not copying. Constants are right in only 15/205 (12 of them are
variant 0005). 8 of 15 validation families still emit one output for every
constant.

## Conclusion

Removing ABI plumbing substantially helps operator binding (87 to 174/205),
syntax (188 to 203 compiled) and the first-argument choice (42 to 100), at the same data
and update count. This supports the noise/length explanation for those parts of
the failure. It does **not** fix literal copying or argument-role binding, so
independent semantic generalization is still unmet; no checkpoint is promoted.
Caveats: single seed, compact BPE relearned, in-lane ablation only against the
SSA lane (no assembly baseline), compact lane limited to straight-line pure
functions, original nested 130-function task and DecBench not measured.

Next candidates, none yet run: why constants collapse to a fixed value
(train-set reproduction check of this checkpoint; literal tokenization, e.g.
digit-level or decimal-rendered literals in the prompt), and a controlled
assembly-text baseline.
