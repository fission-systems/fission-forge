# Fixed-LR GPU continuation: measured regression

On Colab A100-SXM4 40 GB, resume the actual 5,380,608-parameter step-480
checkpoint for 1920 additional updates. Final step 2400, five total epochs,
F32, LR 0.001, seed 42, batch 1, attention chunk 256 and full validation every
120 updates. Corpus lock and input/optimizer provenance remained unchanged.

| Result | Baseline step 480 | Continued step 2400 |
| --- | ---: | ---: |
| Validation NLL | 2.546526410 | 3.151724867 |
| Raw C compiled / all targets | 0 / 160 | 0 / 160 |
| All 64 probes match / all targets | 0 / 160 | 0 / 160 |
| Targets reaching execution | 0 | 0 |

This is a **regression**, not a successful quality improvement. All 320 raw
completions failed compilation; no model candidate executed. The evaluator
separately recompiled original owned C and checked all **10,240 reference
comparisons**. These are not 10,240 failed model executions or a DecBench score.
The two holdout generator families represent 80 source programs with two
optimization variants, not 160 independent algorithms.

Measured training phase: 549.444 seconds, sampled whole-GPU peak 7,266 MiB.
The complete remote driver took 1278.080 seconds, including the CUDA build,
both full generation arms and packaging. Model generation used CUDA, greedy
EOS/byte decoding, 256 new tokens maximum, with every failure retained in the
full denominator. Raw C was not edited or repaired before compilation.

`functional-report.json` records the reference protocol hash, exact driver and
scorer hashes, checkpoint hashes and full aggregation. Both completions files
contain lossless base64 bytes for every target. Generation index files bind
every prompt to its expected checkpoint. The bounded compiler/runtime contract
is reused from the original frozen functional protocol; this is a new GPU
comparison and does not modify that earlier CPU experiment.

`executed-driver.py` is the exact script used for this run: its SHA is verified
against `source.json`. Source archive per-file hashes and dirty-tree provenance
are recorded there. The current driver also supports a later explicit LR fork;
that addition was not executed in this experiment. Downloaded result archive
and final checkpoint SHA were independently verified. Heavy weights remain in
ignored `outputs/colab-continuation-v2/results/continuation/checkpoints/step-00002400/`.

An initial monolithic 53 MiB source/checkpoint upload timed out before training;
the owned runtime was released. The successful attempt uploaded 4 MiB parts,
assembled remotely and verified the archive SHA before launch. Both baseline
and continued generation included all 160 validation inputs. Test model inputs
and external training rows: zero. The successful runtime was stopped after
verified result recovery.

Because fixed-LR continuation regressed, the next separately declared bounded
experiment uses the validation-NLL best historical step-120 checkpoint and an
explicit LR change to 0.0001. It retains Adam history instead of editing the
parent artifact. See [scaling plan](../../docs/scaling.md).
