"""Compare raw P-code from Ghidra (DumpPcode.java) with the corpus's Fission dumps.

Operands are normalized: space names, constants as (value, size), the LOAD/STORE
space-id operand masked, unique temporaries renumbered by first appearance.
Instruction addresses must match exactly. Diagnostic only; nothing is trained on.
"""
import argparse
import collections
import json
from pathlib import Path

SPACES = {0: 'const', 2: 'unique', 3: 'ram', 4: 'register'}


def fission_ops(raw):
    ops = []
    for block in raw['pcode']['blocks']:
        for op in block['ops']:
            def vn(v):
                if v is None:
                    return None
                if v['is_constant']:
                    return ('const', v['constant_val'], v['size'])
                return (SPACES[v['space_id']], v['offset'], v['size'])
            ops.append(dict(address=op['address'], mnemonic=op['asm_mnemonic'], output=vn(op['output']),
                            inputs=[vn(i) for i in op['inputs']]))
    return ops


def ghidra_ops(function):
    def vn(v):
        if v is None:
            return None
        return tuple(v)
    return [dict(address=o['address'], mnemonic=o['mnemonic'], output=vn(o['output']), inputs=[vn(i) for i in o['inputs']])
            for o in function['ops']]


def normalize(ops):
    unique = {}
    out = []
    for op in ops:
        def canon(v, mask=False):
            if v is None:
                return None
            if mask:
                return ('space-id',)
            if v[0] == 'unique':
                return ('unique', unique.setdefault(v[1], len(unique)), v[2])
            return tuple(v)
        memory = op['mnemonic'] in ('LOAD', 'STORE')
        out.append((op['address'], op['mnemonic'], canon(op['output']),
                    tuple(canon(v, mask=memory and i == 0) for i, v in enumerate(op['inputs']))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True, help='full-SSA corpus with pool.jsonl')
    p.add_argument('--ghidra', type=Path, required=True, help='DumpPcode.java output directory')
    p.add_argument('--reference-dump', type=Path, help='compare against another DumpPcode-format directory instead of the corpus Fission dumps')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dumps = {}
    for path in a.ghidra.glob('*.json'):
        d = json.loads(path.read_text())
        for name, function in d['functions'].items():
            dumps[name] = function
    rows = [json.loads(l) for l in (a.corpus / 'pool.jsonl').read_text().splitlines()]
    reference = {}
    if a.reference_dump:
        for path in a.reference_dump.glob('*.json'):
            reference.update(json.loads(path.read_text())['functions'])
    results = []
    for row in rows:
        pair = row['pair']
        name = 'corpus_' + pair['id'].rsplit('-', 1)[0]
        family = pair['id'].rsplit('_', 1)[0]
        if a.reference_dump:
            if name not in reference:
                results.append(dict(id=pair['id'], family=family, status='missing_in_reference'))
                continue
            fission = normalize(ghidra_ops(reference[name]))
        else:
            fission = normalize(fission_ops(pair['input']['raw']))
        if name not in dumps:
            results.append(dict(id=pair['id'], family=family, status='missing_in_ghidra'))
            continue
        ghidra = normalize(ghidra_ops(dumps[name]))
        if fission == ghidra:
            status = 'identical'
        elif [o[:2] for o in fission] == [o[:2] for o in ghidra]:
            status = 'same_ops_different_operands'
        elif sorted(o[1] for o in fission) == sorted(o[1] for o in ghidra):
            status = 'same_multiset_different_order_or_operands'
        else:
            status = 'different_ops'
        rec = dict(id=pair['id'], family=family, status=status, fission_ops=len(fission), ghidra_ops=len(ghidra))
        if status != 'identical':
            first = next((i for i, (x, y) in enumerate(zip(fission, ghidra)) if x != y), min(len(fission), len(ghidra)))
            rec['first_difference'] = dict(index=first, fission=fission[first] if first < len(fission) else None,
                                           ghidra=ghidra[first] if first < len(ghidra) else None)
        results.append(rec)
    counts = collections.Counter(r['status'] for r in results)
    by_family = collections.defaultdict(collections.Counter)
    for r in results:
        by_family[r['family']][r['status']] += 1
    mnemonic_gap = collections.Counter()
    for r in results:
        d = r.get('first_difference')
        if d and d['fission'] and d['ghidra']:
            mnemonic_gap[(d['fission'][1], d['ghidra'][1])] += 1
    summary = dict(rows=len(results), statuses=dict(counts),
                   op_count_equal=sum(1 for r in results if r.get('fission_ops') == r.get('ghidra_ops')),
                   by_family={k: dict(v) for k, v in sorted(by_family.items())},
                   most_common_first_mnemonic_pairs=[[list(k), v] for k, v in mnemonic_gap.most_common(10)])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.output / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
    print(json.dumps({k: summary[k] for k in ('rows', 'statuses', 'op_count_equal', 'most_common_first_mnemonic_pairs')}, indent=2))


if __name__ == '__main__':
    main()
