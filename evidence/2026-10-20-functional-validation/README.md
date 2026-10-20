# Full synthetic validation and an additional-training experiment

**Result: compilation 0/160 and all-probe matches 0/160 in both arms.**
The tiny model has not demonstrated usable C decompilation. This is an owned
synthetic validation measurement, not a DecBench score or real-project result.

## Frozen scope and observed results

[Protocol manifest](protocol-manifest.json) binds the 160 validation pairs
(80 source programs × two optimization inputs), corpus/split/probe/reference
hashes, per-target prompts, compiler/linker, evaluator/worker, Rust executable,
runtime and [predeclared experiment plan](../../configs/evaluation/functional-validation-v1.json).
The baseline checkpoint is step 480. The experiment resumes it for exactly
1,440 train-only updates and selects the predeclared final step 1,920. This is
an attempted improvement through additional training, **not a successful
improvement**. No test input or functional test score selected either checkpoint.

| Measure | Baseline, step 480 | After additional training, step 1,920 |
| --- | ---: | ---: |
| Expected / completed targets | 160 / 160 | 160 / 160 |
| Compiled and linked | 0 / 160 | 0 / 160 |
| All 64 inputs matched | 0 / 160 | 0 / 160 |
| Targets reaching candidate execution | 0 | 0 |
| Candidate input cases actually executed | 0 | 0 |
| Matched / full expected input denominator | 0 / 10,240 | 0 / 10,240 |
| Compilation failures | 160 | 160 |
| Teacher-forced validation NLL | 2.176332 | 2.703266 |

All 320 generations stopped at EOS. Every completion was preserved unchanged
and attempted as raw C; none compiled, so **no model-produced candidate reached
execution**. Their failure rows remain in the full denominator. The zero match
rate does not mean 10,240 candidate executions returned incorrect values.
A sample pair of raw outputs for the same validation target is:

```text
baseline: un ign }
after:    unsignsinsign un b }
```

These are model outputs, not source examples inserted into prompts or repaired
by the evaluator. Full bytes are in [baseline completions](baseline-completions.jsonl)
and [additional-training completions](improved-completions.jsonl); all IDs,
status categories, compiler diagnostics, process exit/deadline records and hashes
are preserved in [baseline records](baseline-records.jsonl) and
[additional-training records](additional-training-records.jsonl).

[Comparison](comparison.json), [baseline report](baseline-report.json), and
[additional-training evaluation report](additional-training-report.json) retain
both checkpoint hashes and the identical protocol identity. There were zero
generation/UTF-8/contract/compile-timeout/ABI/import/execution failures besides
the observed syntax/compilation failures. Later execution stages were never reached by model candidates. Those other failure paths are tested
by controlled fixtures, not counted as extra model targets.

## Reference execution and isolation

Before scoring either arm, all 160 original C variants were compiled to
freestanding Wasm32 and run against all 64 inputs each. Their 10,240 returned
values matched the frozen oracle previously checked against host-native C.
[Inputs](inputs.json), [reference vectors](references.json), and
[the source recipe](corpus-config.json) are retained.

Candidate modules, if compilable, run in Wasmtime with no imports, no WASI or
host functions; every probe gets a fresh instance. Guest memory is capped at
1 MiB, Wasm stack at 256 KiB, and fuel at 100,000 per input, with CPU and wall
limits around compilation/generation/execution. UBSan traps are enabled at
compilation. This is the current uint32/Wasm32 domain, not arbitrary native C
or formal equivalence; finite probes and sanitizer coverage have limits.

The runtime is exercised by passing correct C, wrong-result C, nonterminating C,
division-by-zero C, wrong-ABI modules, imported-host-function modules, oversized
memory, invalid source/encoding and a forced process timeout. Positive arithmetic
fixtures match exactly; negative fixtures produce their expected categories.
Missing denominator rows and modified candidate bytes are also rejected.
[Evaluation test log](evaluation-tests.log) records these actual controls.

## Training experiment and interpretation

[Training report](training-report.json) records the same 70,512-parameter
Rust/Candle model, source corpus lock, F32 optimizer and step-480 resume point.
All 480 training IDs were visited exactly three additional times; validation
IDs never enter the sampled training list. The final global step is 1,920,
with epoch three / cursor 480. Test model examples remain zero.

The [teacher-forced validation curve](validation-curve.json) initially improved to 2.049028 at step 720, then
rose to 2.703266 at step 1,920. The predeclared final-step functional evaluation
was retained rather than replaced after seeing the result. Additional training
did not improve the measured compilation or all-case match rates. Falling
training loss alone does not establish functional recovery, and the worsening
held-out loss is consistent with overfitting in this tiny experiment.

The additional training plus periodic validation/checkpoint writes took about
117 seconds on CPU. No GPU runtime was allocated. Checkpoints and complete local
run trees remain ignored in `outputs/`; they are not published model releases.

## Reproduction, replay and verification

[Commands and contracts](../../docs/functional-evaluation.md) describe freezing,
full generation/evaluation, the exact additional-training run, comparison and
replay. The Rust generator now uses detached KV inference; multi-layer cache
logits and context boundaries are compared with full-prefix inference in tests.
The token policy remains deterministic greedy EOS/byte selection.

A repeated baseline generated identical raw bytes for all 160 targets. The
comparison verifier checks full ID order, record hashes, unchanged source bytes,
all result aggregates, identical protocol and three-epoch train-only coverage.
A separate strict [published-output replay](replay-report.json) regenerated
original C, matched all 10,240 reference executions again, and reproduced both
arms from all 320 public raw outputs without checkpoints, Fission or GPU.
Replay checks compiler/runtime/evaluator identities and artifact hashes.

[Verification](verification.json): 22 Rust tests, eight corpus Python tests,
nine evaluation Python tests (39 total), rustfmt, Python compilation, Clippy
with warnings denied, and the complete published-artifact replay passed.
[Rust tests](rust-tests.log), [corpus tests](corpus-tests.log), and
[Clippy](clippy.log) are recorded.

Runs preceded the publishing commit; verification identifies base `38a9563`
plus a dirty tree, and the manifest identifies exact executable/evaluator hashes.
The base revision alone does not contain these new features. Reproducing model
inference requires the matching local checkpoints or a recreated training run;
published raw C replay reproduces compilation/execution measurements separately.
