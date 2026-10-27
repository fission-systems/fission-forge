"""Does a checkpoint reproduce literal constants of rows it was trained on?

Reads one split of a frozen corpus, compiles each prompt with the corpus lane,
generates with the CPU `forge generate` path and compares the returned
expression with the label. Diagnostic only: no scoring, no checkpoint selection.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import tempfile


def expression(source):
    m = re.search(r'return (.*);', source)
    return m.group(1) if m else ''


def literals(source):
    return re.findall(r'(\d+)u', expression(source))


def skeleton(source):
    return re.sub(r'\d+u', 'K', expression(source))


def generate(forge, checkpoint, lane, row, folder, max_new_tokens):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'raw.json').write_text(json.dumps(row['input']['raw']))
    (folder / 'profile.json').write_text(json.dumps(row['input']['profile']))
    subprocess.run([forge, 'pcode-compile', '--raw', folder / 'raw.json', '--profile', folder / 'profile.json',
                    '--representation', lane, '--output', folder / 'compiled'], check=True, capture_output=True)
    subprocess.run([forge, 'generate', '--checkpoint', checkpoint, '--prompt', folder / 'compiled/prompt.txt',
                    '--representation', lane, '--max-new-tokens', str(max_new_tokens), '--output', folder / 'out'],
                   check=True, capture_output=True)
    return (folder / 'out/completion.txt').read_text(errors='replace')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation'], default='train')
    p.add_argument('--forge', type=Path, default=Path('target/release/forge'))
    p.add_argument('--stride', type=int, default=1, help='Use every Nth row, deterministically')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    lane = json.loads((a.corpus / 'config.json').read_text())['representation']
    rows = [json.loads(l) for l in (a.corpus / 'release' / f'{a.split}.jsonl').read_text().splitlines()][::a.stride]
    records = []
    for row in rows:
        with tempfile.TemporaryDirectory() as temp:
            text = generate(a.forge.resolve(), a.checkpoint.resolve(), lane, row, Path(temp), a.max_new_tokens)
        want, got = literals(row['source']), literals(text)
        records.append(dict(id=row['id'], source=expression(row['source']), generated=expression(text),
                            exact=text.strip() == row['source'].strip(),
                            skeleton_match=skeleton(text) == skeleton(row['source']),
                            literals_match=sorted(want) == sorted(got), generated_literals=got, true_literals=want))
    n = len(records)
    by_variant = collections.defaultdict(lambda: [0, 0])
    for r in records:
        group = 'base_2_15' if int(r['id'].rsplit('_', 1)[1].split('-')[0]) < 14 else 'wide'
        by_variant[group][0] += r['literals_match']
        by_variant[group][1] += 1
    constants = collections.Counter(tuple(r['generated_literals']) for r in records)
    summary = dict(split=a.split, rows=n, exact=sum(r['exact'] for r in records),
                   skeleton_match=sum(r['skeleton_match'] for r in records),
                   literals_match=sum(r['literals_match'] for r in records),
                   literals_match_by_variant_group={k: dict(match=v[0], rows=v[1]) for k, v in by_variant.items()},
                   distinct_generated_expressions=len({r['generated'] for r in records}),
                   most_common_generated_literals=[[list(k), v] for k, v in constants.most_common(8)])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.output / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
