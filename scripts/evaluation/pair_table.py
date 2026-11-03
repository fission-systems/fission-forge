"""Per-operator-pair functional results from score_gpu.py / score_external.py output folders.

Each argument is NAME=FOLDER:ARM, where FOLDER holds one `<id>/<ARM>/result.json` per target.
Rows are grouped by the `rf_*_<outer>_<inner>_*` pair in the target id. Diagnostic summary only.
"""
import argparse
import collections
import json
from pathlib import Path


def pair_of(target_id):
    parts = target_id.split('_')
    return parts[2] + '∘' + parts[3]


def table(folder, arm):
    pairs = collections.defaultdict(lambda: [0, 0])
    totals = [0, 0, 0]
    for result in sorted(Path(folder).glob(f'*/{arm}/result.json')):
        r = json.loads(result.read_text())
        pair = pairs[pair_of(result.parent.parent.name)]
        pair[0] += r['all_cases_match']
        pair[1] += 1
        totals[0] += r['compiled']
        totals[1] += r['all_cases_match']
        totals[2] += r['matched_cases']
    return totals, {k: v for k, v in sorted(pairs.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    out = {}
    for spec in a.runs:
        name, rest = spec.split('=', 1)
        folder, arm = rest.rsplit(':', 1)
        out[name] = table(folder, arm)
    names = list(out)
    all_pairs = sorted({k for _, pairs in out.values() for k in pairs})
    print('%-10s' % 'pair' + ''.join('%22s' % n for n in names))
    for pair in all_pairs:
        print('%-10s' % pair + ''.join('%22s' % ('%d/%d' % tuple(out[n][1].get(pair, [0, 0]))) for n in names))
    print('%-10s' % 'compiled' + ''.join('%22d' % out[n][0][0] for n in names))
    print('%-10s' % 'matched' + ''.join('%22d' % out[n][0][1] for n in names))
    print('%-10s' % 'cases' + ''.join('%22d' % out[n][0][2] for n in names))
    if a.output:
        a.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
