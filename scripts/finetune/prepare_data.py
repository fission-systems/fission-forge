"""Render frozen-corpus prompts for fine-tuning a pretrained causal LM.

Uses the same lane prompt that Forge trains on (`forge pcode-compile`) so only the
model and tokenizer differ. Test rows are never read.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def render(forge, lane, row):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        (temp / 'raw.json').write_text(json.dumps(row['input']['raw']))
        (temp / 'profile.json').write_text(json.dumps(row['input']['profile']))
        subprocess.run([forge, 'pcode-compile', '--raw', temp / 'raw.json', '--profile', temp / 'profile.json',
                        '--representation', lane, '--output', temp / 'out'], check=True, capture_output=True)
        return (temp / 'out/prompt.txt').read_text()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--forge', type=Path, default=Path('target/release/forge'))
    p.add_argument('--probe-stride', type=int, default=3, help='Every Nth training row is also used for the reproduction probe')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    lane = json.loads((a.corpus / 'config.json').read_text())['representation']
    forge = a.forge.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, lane=lane, corpus_lock_sha256=sha((a.corpus / 'release/lock.json').read_bytes()), files={})
    for split in ['train', 'validation']:
        rows = [json.loads(l) for l in (a.corpus / 'release' / f'{split}.jsonl').read_text().splitlines()]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            prompts = list(pool.map(lambda r: render(forge, lane, r), rows))
        text = ''.join(json.dumps(dict(id=r['id'], prompt=prompt, source=r['source']), separators=(',', ':')) + '\n'
                       for r, prompt in zip(rows, prompts))
        (a.output / f'{split}.jsonl').write_text(text)
        manifest['files'][f'{split}.jsonl'] = dict(rows=len(rows), sha256=sha(text.encode()))
    train = [json.loads(l) for l in (a.output / 'train.jsonl').read_text().splitlines()]
    probe = ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in train[::a.probe_stride])
    (a.output / 'train-probe.jsonl').write_text(probe)
    manifest['files']['train-probe.jsonl'] = dict(rows=len(train[::a.probe_stride]), sha256=sha(probe.encode()))
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
