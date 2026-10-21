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

## Continue a real checkpoint and compare generated C

```sh
python3 scripts/colab/run_corpus.py \
  --corpus outputs/corpus-bootstrap-v1 \
  --resume outputs/colab-corpus-gpu-v3/results/pilot/checkpoints/step-00000480 \
  --additional-steps 1920 --output outputs/colab-continuation-new \
  --session forge-continue-new --gpu A100

outputs/eval-venv/bin/python scripts/evaluation/score_gpu.py \
  --results outputs/colab-continuation-new/results \
  --corpus outputs/corpus-bootstrap-v1 --output outputs/colab-functional-new
```

Continuation verifies schema-4 weights and corpus-lock identity before allocation.
It retains the existing seed, LR, batch and validation/checkpoint cadence; the
Rust trainer refuses incompatible resume state. At most 1,920 additional updates
are accepted per bounded job. Baseline and fixed final checkpoint each generate
all 160 validation prompts with a 256-token greedy budget on CUDA. Reference C
and expected values are used only by the separate local scorer. Generation
errors/timeouts remain in the full denominator. The scorer verifies prompt,
checkpoint and byte identity and rechecks original C on all 10,240 reference
cases before reporting compile and probe-match rates. It does not change the
old CPU functional protocol or claim a DecBench score.

Archives above 8 MiB upload in 4 MiB parts and are assembled remotely with a
full SHA-256 check before launch. Continuation result archives retain final weights and any validation-best
checkpoint produced in that job, plus intermediate state/report metadata.
A historical best from another run must be retained separately; its metadata
can remain present even when its original absolute path is unavailable.
Other intermediate weights remain only on the temporary VM and are not exported.
If a phase fails before a final report exists, partial checkpoints are retained
for recovery. Cleanup still releases the named runtime after artifact collection.

A relabelled project license does not rewrite historical frozen corpus metadata.
The current owned corpus lock remains unchanged for checkpoint compatibility;
newly generated owned-source corpora record Apache-2.0.

## Explicit learning-rate fork

A changed LR is rejected on ordinary resume. For a declared new experiment:

```sh
python3 scripts/colab/run_corpus.py \
  --corpus outputs/corpus-bootstrap-v1 \
  --resume outputs/colab-corpus-gpu-v3/results/pilot/checkpoints/step-00000120 \
  --resume-learning-rate 0.0001 --additional-steps 960 \
  --output outputs/colab-lr-recovery-new --session forge-lr-new --gpu A100
```

The controller declares the change in `job.json`. The Rust CLI uses
`--override-learning-rate` with `--resume` and the new `--learning-rate`.
Weights, Adam moments, optimizer step, seed and sampler are loaded unchanged;
only LR is replaced. The parent checkpoint is never edited. The report records
old/new LR and the parent checkpoint SHA/step. Subsequent ordinary resumes use
the newly saved LR. Model, corpus and validation-setting mismatches remain
errors even with the LR override. A real CPU corpus integration test exercises
rejection, explicit override, unchanged parent hashes and subsequent resume.

Freshly rebuilt Apache-2.0 corpus metadata changes the release lock and cannot
resume an old checkpoint expecting the historical lock. Preserve the old frozen
corpus to reproduce these recorded runs. A new corpus needs a newly declared
functional plan with its own lock/checkpoint hashes and a fresh
`functional.py freeze` run. To use that fresh freeze with `score_gpu.py --reference-evidence`,
copy its `manifest.json` to `protocol-manifest.json` inside the same new freeze
directory; keep its generated `inputs.json` and `references.json`. Do not edit
an old frozen manifest to make a new corpus appear compatible.

## Fresh corpus and BPE trials

For a newly frozen corpus, `--fresh-epochs 1..8` initializes seed-42 weights
and trains complete shuffled epochs at `--fresh-learning-rate` (default 0.0003).
It refuses an old resume checkpoint. Full validation and saved checkpoints run
every 256 updates and at the final update; the in-job NLL best is selected before
all validation prompts are generated. A longest-encoded-row one-update preflight
checks actual sampler identity and the 85% whole-device memory gate. Final and
NLL-best weights are downloaded; other intermediate tensor payloads are omitted.

`--tokenizer PATH` adds an explicitly trained lexical BPE specification to a
fresh job, validates its training-split hash and adapts the model vocabulary.
The Rust trainer independently checks codec identity and context fit. Generation
uses the checkpoint's embedded codec and caps both newly emitted tokens and
raw bytes at 512. The original bootstrap resume pilot still uses its byte codec.
See the [composition corpus and frozen evaluation plans](composition-corpus.md)
for exact data counts, restrictions, failure records and reproduction commands.

The local wrapper accepts schema-4 byte checkpoints for the historical
continuation mode. Schema-5 BPE training can resume directly through the Rust
`train-corpus --tokenizer ... --resume ...` interface; the wrapper's
`--resume` mode is not yet a BPE continuation interface.
