# HF discovery and staging, 2026-10-20

`inventory.json` records seven pinned Hub metadata/card inspections and bounded
HEAD Viewer previews. File listings are reduced to counts/examples; dataset
source rows and upstream cards are not redistributed here.

`exebench-summary.json` and `exebench-records.jsonl` describe a 100-row prefix of
the pinned original ExeBench archive. The CLI verified the expected content SHA
before reading it. Records contain source hashes/provenance and eligibility
signals, not source code. All rows remain `training_approved: false`.

See [intake documentation](../../docs/hf-datasets.md) for commands, priorities,
license/provenance gaps and benchmark exclusions. No external source code was
executed and zero external rows were admitted to model training.
