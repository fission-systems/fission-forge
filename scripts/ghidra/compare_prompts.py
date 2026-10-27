"""Does the model-facing prompt survive a change of lifter?

Converts Ghidra raw P-code (DumpPcode.java) into Forge's raw-dump schema for
straight-line functions, compiles it with the same lane as the corpus and compares
the prompt text with the Fission-derived one. The Fission profile is reused for the
compiler, so only the P-code differs. LOAD/STORE space ids are mapped to the
profile's ram id; opcode spellings come from the Fission corpus's own mnemonic table.
"""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import tempfile

SPACE_IDS = {'unique': 2, 'ram': 3, 'register': 4}


def convert(function, mnemonic_to_opcode):
    ops = []
    for seq, op in enumerate(function['ops']):
        def vn(v, memory_space=False):
            if v is None:
                return None
            space, offset, size = v
            if space == 'const':
                if memory_space:
                    return dict(space_id=0, offset=3, size=size, is_constant=True, constant_val=3)
                # Ghidra reports the constant as a signed long; Forge stores the unsigned,
                # size-masked offset next to the signed value.
                return dict(space_id=0, offset=offset & ((1 << (size * 8)) - 1), size=size, is_constant=True, constant_val=offset)
            return dict(space_id=SPACE_IDS[space], offset=offset, size=size, is_constant=False, constant_val=0)
        memory = op['mnemonic'] in ('LOAD', 'STORE')
        ops.append(dict(seq_num=seq, opcode=mnemonic_to_opcode[op['mnemonic']], address=op['address'],
                        output=vn(op['output']), inputs=[vn(v, memory and i == 0) for i, v in enumerate(op['inputs'])],
                        asm_mnemonic=op['mnemonic']))
    return dict(entry_address=hex(function['entry']), decode_address=hex(function['entry']),
                decoded_instructions=len({o['address'] for o in function['ops']}), stop_reason='terminal_control_flow',
                template_source_counts={'sla_construct_tpl': len({o['address'] for o in function['ops']})},
                pcode=dict(blocks=[dict(index=0, start_address=function['entry'], successors=[], ops=ops)]))


def compile_prompt(forge, raw, profile, lane, folder):
    (folder / 'raw.json').write_text(json.dumps(raw))
    (folder / 'profile.json').write_text(json.dumps(profile))
    result = subprocess.run([forge, 'pcode-compile', '--raw', folder / 'raw.json', '--profile', folder / 'profile.json',
                             '--representation', lane, '--output', folder / 'out'], capture_output=True, text=True)
    if result.returncode:
        return None, result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'failed'
    return (folder / 'out/prompt.txt').read_text(), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True, help='full-SSA corpus with pool.jsonl')
    p.add_argument('--ghidra', type=Path, required=True)
    p.add_argument('--forge', type=Path, default=Path('target/release/forge'))
    p.add_argument('--lanes', nargs='+', default=['pcode-ssa-return-u32-compact-v1', 'pcode-ssa-return-u32-compact-dec-v1', 'pcode-plus-ssa-return-u32-v1'])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    forge = a.forge.resolve()
    dumps = {}
    for path in a.ghidra.glob('*.json'):
        dumps.update(json.loads(path.read_text())['functions'])
    rows = [json.loads(l)['pair'] for l in (a.corpus / 'pool.jsonl').read_text().splitlines()]
    table = collections.defaultdict(set)
    for pair in rows:
        for block in pair['input']['raw']['pcode']['blocks']:
            for op in block['ops']:
                table[op['asm_mnemonic']].add(op['opcode'])
    mnemonic_to_opcode = {k: next(iter(v)) for k, v in table.items()}
    out = collections.defaultdict(lambda: collections.Counter())
    examples = {}
    for pair in rows:
        name = 'corpus_' + pair['id'].rsplit('-', 1)[0]
        function = dumps.get(name)
        for lane in a.lanes:
            if function is None:
                out[lane]['missing_in_ghidra'] += 1
                continue
            if any(op['mnemonic'] not in mnemonic_to_opcode for op in function['ops']):
                out[lane]['unmapped_ghidra_opcode'] += 1
                continue
            with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
                fission, e1 = compile_prompt(forge, pair['input']['raw'], pair['input']['profile'], lane, Path(t1))
                ghidra, e2 = compile_prompt(forge, convert(function, mnemonic_to_opcode), pair['input']['profile'], lane, Path(t2))
            if fission is None:
                out[lane]['fission_rejected'] += 1
            elif ghidra is None:
                out[lane]['ghidra_rejected'] += 1
                examples.setdefault((lane, 'ghidra_rejected'), (pair['id'], e2))
            elif fission == ghidra:
                out[lane]['identical_prompt'] += 1
            else:
                out[lane]['different_prompt'] += 1
                examples.setdefault((lane, 'different_prompt'), (pair['id'], fission, ghidra))
    summary = dict(rows=len(rows), lanes={lane: dict(c) for lane, c in out.items()},
                   examples={f'{k[0]}|{k[1]}': v for k, v in examples.items()})
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
