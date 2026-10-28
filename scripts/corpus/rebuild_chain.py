"""Rebuild the shallow-wide-v3 corpus chain and verify it against committed evidence.

Builds only into paths that do not exist yet (it never deletes anything), then compares
the release file digests and the tokenizer digest with the locks stored in `evidence/`.
A mismatch means the toolchain (compiler, lifter, SLA) is not the one that produced the
recorded run, so recorded numbers must not be attributed to the rebuilt data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CHAIN = [
    dict(name='return-v3', lane='pcode-plus-ssa-return-u32-v1', isolate=False,
         lock='evidence/2026-10-26-wide-constant/corpus-lock.json', tokenizer='evidence/2026-10-26-wide-constant/tokenizer.json'),
    dict(name='compact-v3', lane='pcode-ssa-return-u32-compact-v1', isolate=False,
         lock='evidence/2026-10-28-compact-lane/corpus-lock.json', tokenizer='evidence/2026-10-28-compact-lane/tokenizer.json'),
    dict(name='compactdec-v3', lane='pcode-ssa-return-u32-compact-dec-v1', isolate=True,
         lock='evidence/2026-10-28-decimal-digit-literals/corpus-lock.json', tokenizer='evidence/2026-10-28-decimal-digit-literals/tokenizer.json'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=Path('outputs'))
    p.add_argument('--forge', type=Path, default=Path('target/release/forge'))
    p.add_argument('--cc', default='/opt/homebrew/opt/llvm/bin/clang')
    p.add_argument('--nm', default='/opt/homebrew/opt/llvm/bin/llvm-nm')
    a = p.parse_args()
    full = a.outputs / 'corpus-shallow-wide-full-v3'
    if full.exists():
        sys.exit(f'refusing to overwrite {full}')
    run([sys.executable, 'scripts/corpus/build.py', '--config', 'configs/corpus/shallow-wide-v3.json', '--output', full,
         '--forge', a.forge, '--cc', a.cc, '--nm', a.nm])
    report = {}
    for step in CHAIN:
        corpus = a.outputs / f"corpus-shallow-wide-{step['name']}"
        tok = a.outputs / f"tokenizer-shallow-wide-{step['name']}"
        for path in (corpus, tok):
            if path.exists():
                sys.exit(f'refusing to overwrite {path}')
        run([sys.executable, 'scripts/corpus/project_return.py', '--corpus', full, '--output', corpus, '--forge', a.forge, '--lane', step['lane']])
        cmd = [a.forge, 'tokenizer-train', '--corpus', corpus, '--vocab-size', 772, '--output', tok]
        if step['isolate']:
            cmd.append('--isolate-digits')
        run(cmd)
        expected = json.loads(Path(step['lock']).read_text())['files']
        got = {name: sha(corpus / 'release' / name) for name in expected}
        report[step['name']] = dict(release_files_match=got == expected,
                                    mismatched=[n for n in expected if got[n] != expected[n]],
                                    tokenizer_matches=sha(tok / 'tokenizer.json') == sha(step['tokenizer']))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
