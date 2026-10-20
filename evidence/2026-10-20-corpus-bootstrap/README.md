# Owned synthetic bootstrap corpus — 2026-10-20 KST

This records **data construction and integrity checks**, not corpus training,
held-out model accuracy, real-project generalization, or DecBench scoring.
No GPU runtime was allocated and no model/dataset was published to Hugging Face.

## Measured output

The fixed [recipe](../../configs/corpus/bootstrap-v1.json) generated 400 owned C
programs from ten templates, 40 constant variants each. A host-native build
matched a separate wrapping-u32 oracle on all **25,600 comparisons** (64 per
program). The generated x86-64 ELF binaries were statically inspected only.
This does not establish runtime parity between those binaries and their IR.

| Split | Generator families | Source programs | Attempted/supported | Released pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 6 | 240 | 720 / 720 | 480 |
| Validation | 2 | 80 | 240 / 240 | 160 |
| Test | 2 | 80 | 240 / 240 | 160 |
| Total | 10 | 400 | 1,200 / 1,200 | 800 |

Every `O0/O1/O2` variant was attempted. All 1,200 passed strict P-code+ SSA v1
conversion in this deliberately restricted compiler/template distribution.
For every source, `O1/O2` produced identical normalized prompt/label pairs;
400 duplicate `O2` rows were excluded. No cross-split fingerprint component or
ambiguous prompt/label was observed. The full [audit](audit.json) retains every
excluded row and split context counts. This is 400 source programs from ten
templates, not 1,200 independent algorithms.

Splits were assigned by generator family before compilation. Validation uses
remainder/xor_fold and test uses clamp/quotient_round; none are training families.
These are synthetic family-disjoint sets, **not independent real projects**.
DecBench source data was not used; actual DecBench overlap remains unmeasured.

## Provenance and integrity

[Manifest](manifest.json) binds the recipe, generator hash, compiler version and
flags, pinned Fission/SLA profile, Forge executable, source files, native check,
ELF files, program records, all attempts, and supported pool by SHA-256.
[Release lock](release-lock.json) binds the manifest, split files, and audit.
The build ran with uncommitted corpus features on base `5b8bb4d`; the manifest
correctly records a dirty tree and the exact executable/generator hashes. The
base revision alone does not contain the corpus implementation.

Full generated artifacts remain locally in ignored
`outputs/corpus-bootstrap-v1/`; they are reproducible with the
[documented commands](../../docs/corpus.md). The large JSONL/raw/binary outputs
are not committed. [Build log](build.log) retains extraction progress and totals;
[native verification](native-verification.json) records source-label scope.

A separate 20-program / 60-variant real builder output exercised integration
regressions on temporary copies: altered frozen validation bytes, modified
artifacts, rehashed split swaps, rehashed incorrect labels, missing attempt
records, cross-split clone quarantine leaving an empty test set, and retention
of deliberately unsupported attempts in the denominator. All expected failures
were detected, and the unsupported case could finalize with its denominator
intact. Rust also tests transitive family quarantine and malformed ELF rejection.
The full 1,200-attempt release then passed a read-only re-audit.

[Verification record](verification.json): 18 Rust tests, eight corpus Python
tests (including integration tests), two existing evaluation Python tests,
format check and Clippy with warnings denied. The earlier IR execution tests
remain bounded to the owned diamond/loop cases, not this entire corpus.

## Model/training boundary

The existing 61,888-parameter model and greedy C generation already work on a
memorized fixture. The approximately 100M configuration previously passed a
short A100 training/checkpoint probe. **Neither is a corpus-pretrained,
generalizing decompiler. No new checkpoint was trained here.**

Released shifted sequences range from 1,167 to 5,321 byte tokens. All 800 fit an
8,192-token budget, while none fit the original 1,024-token IR probe. The real
trainer [refused a corpus row](trainer-boundary.json) rather than truncating it;
no checkpoint/output directory was created. Its existing 32-example fixed-batch
limit and lack of a corpus sampler remain explicit. Longer-context GPU capacity,
sampler-aware resumption, train-only sampling, held-out loss/generation, and a
final untouched test evaluation are the next training work.
