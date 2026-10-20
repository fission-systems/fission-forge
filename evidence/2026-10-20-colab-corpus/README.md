# Colab corpus GPU pilot, 2026-10-20 KST

A real A100-SXM4 40 GB ran the Rust/Candle F32 corpus trainer from random
initialization. The audited owned synthetic corpus was used; no HF data or
pretrained weights entered training.

| Measurement | Result |
| --- | --- |
| Model | 5,380,608 parameters; 4 layers, width 256, 4 heads, FF 1024, context 8192 |
| Longest training input exercised | 4,878 tokens |
| Training | 480 updates, batch 1, one complete epoch, seed 42, LR 0.001 |
| Validation | All 160 independent validation pairs, completion-token weighted |
| Initial / final validation NLL | 5.529699742 → 2.546526410 |
| Pilot wall time | 148.608 s, including startup and initial/full periodic validation |
| Training + periodic validation | 132.107 s |
| Sampled whole-device peak | 7,266 MiB, 0.2 s sampling; transient peaks may be missed |
| Test model batches | 0 |
| Resume check | One + resumed one matches uninterrupted two: weights and Adam byte-identical, sampler identical |
| GPU inference | One held-out prompt, greedy, stopped at EOS after 28 tokens |
| Functional / DecBench score | Not measured for this model |

The generated sample (`generation-completion.txt`) is **not valid C**. Lower NLL
and successful CUDA execution do not establish decompilation quality. This model
has not demonstrated improvement over the earlier tiny CPU baseline, and this
measurement does not establish long-input 100M/300M training capacity.

`source.json` records exact uploaded file hashes, the prior HEAD and dirty-tree
flag, corpus lock, archive SHA and remote-driver SHA. This pilot was run from the
implementation worktree before commit. The uploaded source and driver hashes
are authoritative: later controller hardening (launch-timeout retry, result
archive verification and richer observation) was tested locally after the run.
The original controller retrieved/stopped successfully; downloaded result archive
integrity was independently verified locally against terminal status.

`pilot-report.json`, preflight reports, resource samples, environment/binary
metadata and terminal status are copied without recomputing their values.
`verification.json` records local artifact/validation-identity/sampler checks and
47 passing tests (22 Rust, 8 corpus, 9 evaluation, 4 controller, 4 intake).
Per-device numerical reproducibility is measured on the recorded CUDA stack;
these byte equality checks are not a cross-version/device guarantee.

Weights and optimizer checkpoints were downloaded to ignored local storage:
`outputs/colab-corpus-gpu-v3/results/pilot/checkpoints/step-00000480/`.
Reports preserve `/content/...` paths from the VM; substitute the corresponding
local path when loading the checkpoint. Heavy weights are not committed or
published to HF. Result archive SHA:
`fbb70a7997cdf4afcbc6f935fd26a739ae7b8c289b978e4797bd4f6dcfef47cf`.

The first setup attempt lost its initial kernel connection; another stopped at
Rust bootstrap after a local controller variable bug. The fixed fresh run
completed. Account balance changed from 246.75 to 246.36 CUs across setup/retries
and the completed run; this is not a universal GPU hourly price. After cleanup,
server-side sessions were zero and usage rate was 0.00/hr (captured here).

[Reproduction and bounded execution](../../docs/colab-corpus.md) ·
[HF candidate research](../../docs/hf-datasets.md)
