# Hugging Face data intake

The GPU pilot uses the audited owned synthetic corpus only. External Hub data is
staged for research, with **zero external rows approved for training**.
The pinned registry is [`hf-candidates-v1.json`](../configs/data/hf-candidates-v1.json).

| Priority | Dataset | Observed contents | Intended use / blocker |
| --- | --- | --- | --- |
| 1 | [Original ExeBench](https://huggingface.co/datasets/jordiae/exebench) | C functions, dependencies, assembly and real/synthetic IO archives | Recompile eligible C, lift our binaries to the pinned P-code+ SSA lane; source licenses and project overlap require review |
| 2 | [Decompile-Bench](https://huggingface.co/datasets/LLM4Binary/decompile-bench) | Viewer HEAD: 1,359,586 rows, `name/code/asm/file`; C and C++ mixed | Recover original source/build provenance; assembly is not our input representation |
| 2 | [Decompile-Bench binaries](https://huggingface.co/datasets/LLM4Binary/decompile-bench-bins) | 71 multipart ZIP files | Potential binary lifting, but function mapping, provenance and download size need investigation; archive not downloaded |
| 3 | [Ghidra 100k](https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k) | Viewer HEAD: 100,000 `instruction/output` pairs | Pseudo-C refinement data, not raw P-code or SSA |
| 3 | [ExeBench IO derivative](https://huggingface.co/datasets/mistral0105/exebench_io_validated_full) | Viewer HEAD: 17,238 rows; source, dependencies, IO, path and ref | Convenient Parquet, but preserve original provenance; no explicit card license observed |
| Reserved | [Noelo DecBench](https://huggingface.co/datasets/noelo-lab/decbench-dataset) | Metadata and card only | Evaluation only; no examples sampled for training research |
| Reserved | [LLM4Binary evaluation](https://huggingface.co/datasets/LLM4Binary/decompile-eval) | Metadata and card only | Evaluation only, even if a viewer uses the split name `train` |

LLM4Binary's **Decompile-Bench is distinct from Noelo DecBench**. A dataset-card
license does not establish every included upstream source file's license.
The Viewer API describes current HEAD, not revision-pinned training content.
Viewer errors for ExeBench and the binary collection do not indicate empty data.
Pinned Hub metadata/cards and bounded HEAD previews are distinguished in the
[recorded inventory](../evidence/2026-10-20-hf-datasets/inventory.json).

## Reproduce metadata and bounded staging

```sh
python3 scripts/data/hf_inventory.py --output outputs/hf-inventory-new
python3 -m pip install -r scripts/data/requirements.txt
hf download jordiae/exebench train_real_simple_io.tar.gz \
  --repo-type dataset --revision 093085f8558cfd53de8e2c8f4ccc7b9e73dc22ae \
  --local-dir outputs/hf-intake/exebench
python3 scripts/data/exebench_probe.py \
  --archive outputs/hf-intake/exebench/train_real_simple_io.tar.gz \
  --expected-sha256 5717bb6c663ab3fe039208511fb8e52e3db7b45b7d357d3e4ca8d027b715b19e \
  --output outputs/hf-intake/exebench-probe-new --limit 100
```

The 35,982,092-byte archive is pinned by revision and content SHA. Its first
100 rows cover 89 project names: all 100 contain real IO pairs, six contain
floating-type keywords, none exposes an explicit source license in `meta`, and
none has an immutable original Git commit in `ref`. This bounded prefix is not
a representative sample or 94 verified trainable examples. IO presence is not
independent execution validation. The probe reads compressed JSON without
extracting members or executing dataset scripts, wrappers, source or binaries.
Committed evidence contains hashes and provenance metadata, not copied C code.

## Admission contract still to implement

1. Resolve source-file/project licenses and attribution; preserve immutable origin.
2. Exclude DecBench/evaluation projects and detect source/binary overlap before
   choosing project-disjoint training and validation sets.
3. Select supported C signatures/dependencies, compile in an isolated bounded
   worker, record compiler/flags/binary hashes and function boundaries.
4. Extract with the pinned Fission/SLA profile; reject incomplete or unsupported
   lifts; validate labels independently with IO tests.
5. Enforce context length, deduplication and frozen corpus audit/release contracts.

No external rows pass these gates yet. Existing synthetic-corpus splitting does
not establish independence from real-world evaluation projects.
