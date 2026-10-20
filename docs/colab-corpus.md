# Colab GPU corpus pilot

The model and optimizer execute in **Rust/Candle on CUDA**. Python packages
explicit source files and the frozen owned corpus, operates the Colab CLI and
collects measurements. No Python ML framework or external weights are used.

## Run

Use an authenticated `google-colab-cli` installation (`colab`), an audited frozen
[bootstrap corpus](corpus.md), and a fresh output/session name. Run from the repo
root with Python 3.12+ (safe tar extraction is required):

```sh
python3 scripts/colab/run_corpus.py \
  --corpus outputs/corpus-bootstrap-v1 \
  --output outputs/colab-corpus-new \
  --session forge-corpus-new --gpu A100
```

`--package-only` checks/packages inputs without allocating a runtime. `--attach`
explicitly reuses the named existing runtime and **still stops it on exit**.
Do not attach another workload's runtime. A100 availability is not guaranteed;
selectability in the UI does not establish allocation or VRAM capacity.
The local process must stay alive until it collects results and stops the VM;
closing/killing it or losing the host can defeat cleanup. Confirm `colab sessions`
after interruption and stop the owned session explicitly if necessary.

The controller uploads source/model configs and audited corpus artifacts only.
Tokens, CLI credentials, `.git`, raw attempt directories and local environment
files are excluded. An exact per-file hash manifest records dirty worktree
content, so a source revision alone is not mistaken for the uploaded source.
The remote driver independently audits the corpus through `train-corpus`.

## Bounded execution contract

1. Install Rust as needed and build the locked Rust/Candle CUDA binary for the
   GPU's compute capability.
2. Use [`gpu-pilot-5m-8k.json`](../configs/model/gpu-pilot-5m-8k.json): 4 layers,
   hidden width 256, 4 heads, FF width 1024, 8192 positions, byte vocabulary 260.
3. Train the longest owned training example (4,878 tokens, seed 619) for one
   update. Refuse the pilot if sampled whole-device memory exceeds 85% of VRAM.
4. Compare a resumed second update against two uninterrupted updates. Require
   byte-identical weights **and Adam moments**, plus matching sampler state.
5. Initialize fresh weights (seed 42), train 480 updates at batch size one,
   attention query chunk 256 and learning rate 0.001. Evaluate all 160 validation
   rows initially and every 120 updates. Test rows never enter model batches.
6. Generate at most 128 C bytes from one held-out prompt on CUDA. This checks
   inference plumbing, not successful compilation or functional equivalence.
7. Package checkpoints, reports and logs, download artifacts, verify the archive
   SHA and release the runtime in the controller's `finally` path.

The pilot has a 900-second training-phase timeout, a 25-minute driver budget
checked during measured phases and a 30-minute controller observation budget.
Build/download/cleanup calls have separate timeouts. Observations can time out
while training continues: the controller retries the same session and uses an
idempotent PID-aware launch rather than starting duplicate jobs.

## Artifacts and limitations

Local output contains `results.tar.gz`, source `payload.json`, terminal status
and `results/`. Checkpoints (weights plus optimizer) are under
`results/pilot/checkpoints/`; reports retain their original `/content/...` paths.
Use the corresponding **local checkpoint path** for local generation/resume.
Checkpoints are ignored locally, not committed or automatically uploaded to HF.
Source archive and result archive hashes must be retained with experimental
records. Rebuilding an unpinned Rust/Colab base environment can change results;
recorded environment versions describe the measured run.

F32, completion-only loss, AdamW and the existing learned-position decoder are
unchanged. Query tiling does not make training activations linear in sequence
length. This 5.38M pilot establishes the measured configuration only; it does not
validate 100M/300M long-context training. Memory polling observes the entire GPU
and can miss transient peaks. NLL reduction is not functional C accuracy or a
DecBench score. External HF data remains subject to [separate intake gates](hf-datasets.md).

Run controller tests without spending compute units:

```sh
python3 -m unittest discover -s scripts/colab -p 'test_*.py'
python3 -m unittest discover -s scripts/data -p 'test_*.py'
```

The fake CLI tests cover result recovery and release, lost observations without
duplicate allocation, idempotent launch retry after timeout, and release after
launch failure. These complement the recorded real GPU run.
