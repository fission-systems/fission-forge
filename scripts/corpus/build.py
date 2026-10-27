"""Build an owned synthetic C/IR bootstrap corpus; Rust audits/finalizes splits.

This is compilation/data orchestration, not model math. Only generated host C is
executed for label checks; x86 ELF targets and benchmark binaries are never run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import expressions
import shallow

MASK = 2**32 - 1
FAMILIES = {'affine', 'bitmask', 'rotate_mix', 'select', 'bounded_loop', 'carry',
            'remainder', 'xor_fold', 'clamp', 'quotient_round'}
FAMILIES |= set(expressions.TEMPLATES) | set(shallow.TEMPLATES) | set(shallow.EXTRA_TEMPLATES)
SPLITS = {'train', 'validation', 'test'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def run(command, **kwargs):
    return subprocess.check_output([str(a) for a in command], text=True,
                                   stderr=subprocess.PIPE, timeout=180, **kwargs)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def config_read(path):
    cfg = json.loads(path.read_text())
    required = {'schema_version', 'name', 'seed', 'variants_per_family', 'optimizations', 'representation', 'families'}
    if not required <= set(cfg) <= required | {'extra_variants'}:
        raise ValueError('unsupported corpus configuration fields')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1 or cfg['representation'] != 'pcode-plus-ssa-v1':
        raise ValueError('unsupported corpus schema/representation')
    if type(cfg['seed']) is not int or type(cfg['variants_per_family']) is not int or not 1 <= cfg['variants_per_family'] <= 256:
        raise ValueError('invalid seed or family size (1..256)')
    if not re.fullmatch('[a-z0-9-]+', cfg['name']):
        raise ValueError('invalid corpus name')
    if not cfg['optimizations'] or len(set(cfg['optimizations'])) != len(cfg['optimizations']) or not set(cfg['optimizations']) <= {'O0', 'O1', 'O2', 'O3'}:
        raise ValueError('unsupported or repeated optimization')
    if not set(cfg['families']) <= FAMILIES or set(cfg['families'].values()) != SPLITS:
        raise ValueError('requires known families and all three explicit splits')
    extra = cfg.get('extra_variants', {})
    if type(extra) is not dict or not set(extra) <= {f for f, split in cfg['families'].items() if split == 'train'}:
        raise ValueError('extra variants are limited to explicit train families')
    if any(type(n) is not int or not 1 <= n <= 256 - cfg['variants_per_family'] for n in extra.values()):
        raise ValueError('invalid extra variant count')
    return cfg


def program(family, index, seed, split):
    rng = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{family}:{index}'.encode()).digest(), 'big'))
    c = rng.getrandbits(32) or 1
    k = (rng.getrandbits(16) | 3)
    m = (rng.getrandbits(32) & 0xfffffffe) or 2
    shift = 1 + rng.randrange(31)
    lo = 1 + rng.randrange(32768)
    hi = lo + 1 + rng.randrange(32768)
    bodies = {
        'affine': f'return (a * {k}u + b) ^ {c}u;',
        'bitmask': f'return ((a ^ {c}u) + b) & {m}u;',
        'rotate_mix': f'unsigned int x = (a << {shift}u) | (a >> {32-shift}u); return x ^ (b + {c}u);',
        'select': f'return a > b ? a + {k}u : b ^ {c}u;',
        'bounded_loop': f'unsigned int x = a; for (unsigned int i = 0; i < (b & 7u); ++i) x = x * {k}u + {c}u; return x;',
        'carry': f'unsigned int s = a + b; return s < a ? {c}u : s ^ {k}u;',
        'remainder': f'unsigned int d = (b & 65535u) + 1u; return (a % d) ^ {c}u;',
        'xor_fold': f'unsigned int x = a ^ {c}u; x ^= x >> {shift}u; x *= {k}u; return x ^ b;',
        'clamp': f'unsigned int x = a; if (x < {lo}u) x = {lo}u; if (x > {hi}u) x = {hi}u; return x ^ b;',
        'quotient_round': f'unsigned int d = (b & 255u) + 1u; return (a / d) + ((a % d) != 0u) + {c}u;',
    }
    identifier = f'{family}_{index:04d}'
    expression = (expressions.instantiate(family,index,seed) if family in expressions.TEMPLATES else shallow.instantiate(family,index,seed) if family in (shallow.TEMPLATES|shallow.EXTRA_TEMPLATES) else None)
    body = 'return '+expressions.render(expression)+';' if expression is not None else bodies[family]
    source = f'unsigned int f(unsigned int a, unsigned int b) {{ {body} }}'
    return dict(id=identifier, family=family, split=split, compiled_name='corpus_' + identifier,
                source=source, parameters=dict(c=c, k=k, m=m, shift=shift, lo=lo, hi=hi), **(dict(expression_ast=expression) if expression is not None else {}))


def oracle(p, a, b):
    if 'expression_ast' in p:return expressions.evaluate(p['expression_ast'],a,b)
    q = p['parameters']; c, k, m, s = q['c'], q['k'], q['m'], q['shift']
    kind = p['family']
    if kind == 'affine': result = ((a*k+b) & MASK) ^ c
    elif kind == 'bitmask': result = ((a ^ c) + b) & m
    elif kind == 'rotate_mix': result = (((a << s) | (a >> (32-s))) & MASK) ^ ((b+c) & MASK)
    elif kind == 'select': result = a+k if a > b else b ^ c
    elif kind == 'bounded_loop':
        result = a
        for _ in range(b & 7): result = (result*k+c) & MASK
    elif kind == 'carry':
        total = (a+b) & MASK
        result = c if total < a else total ^ k
    elif kind == 'remainder': result = (a % ((b & 65535)+1)) ^ c
    elif kind == 'xor_fold':
        x = a ^ c
        result = (((x ^ (x >> s))*k) & MASK) ^ b
    elif kind == 'clamp': result = min(max(a, q['lo']), q['hi']) ^ b
    elif kind == 'quotient_round':
        d = (b & 255)+1
        result = a//d + int(a % d != 0) + c
    else: raise ValueError(kind)
    return result & MASK


def probes():
    boundaries = [0, 1, 2, 7, 0x7fffffff, 0x80000000, MASK]
    result = [(a, b) for a in boundaries for b in boundaries]
    rng = random.Random(7419)
    result += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(15)]
    return result


def canonical_ast(node, erase_literals=False):
    declarations = {node['id']: 'self'}
    def collect(n):
        if n.get('kind') in ('ParmVarDecl', 'VarDecl'):
            declarations[n['id']] = f'd{len(declarations)-1}'
        for child in n.get('inner', []): collect(child)
    collect(node)
    def simplify(n):
        if not n: return {'kind': 'EmptySlot'}
        result = {'kind': n['kind']}
        if 'type' in n: result['type'] = n['type']['qualType']
        for field in ('opcode', 'castKind', 'isPostfix'):
            if field in n: result[field] = n[field]
        if 'value' in n: result['value'] = '#' if erase_literals else n['value']
        if n['kind'] in ('ParmVarDecl', 'VarDecl'): result['declaration'] = declarations[n['id']]
        if 'referencedDecl' in n:
            result['reference'] = declarations[n['referencedDecl']['id']]
        result['inner'] = [simplify(child) for child in n.get('inner', [])]
        return result
    return simplify(node)


def function_bytes(binary, address, size):
    # ELF64 little-endian section headers; target is fixed by the corpus contract.
    if binary[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', binary, 18)[0] != 62:
        raise ValueError('expected ELF64 LE x86-64')
    table = struct.unpack_from('<Q', binary, 40)[0]
    width, count = struct.unpack_from('<HH', binary, 58)
    if width < 64 or size <= 0: raise ValueError('invalid section/symbol bounds')
    for i in range(count):
        off = table+i*width
        flags, start, file_offset, length = struct.unpack_from('<QQQQ', binary, off+8)
        if flags & 4 and start <= address and address+size <= start+length:
            begin = file_offset+address-start
            data = binary[begin:begin+size]
            if len(data) == size: return data
    raise ValueError('function is not contained in an executable section')


def build(args):
    root = Path.cwd()
    output = args.output.resolve()
    if output.exists(): raise ValueError(f'output already exists: {output}')
    cfg = config_read(args.config)
    profile_bytes = args.profile.read_bytes()
    profile = json.loads(profile_bytes)
    fission = args.fission_root.resolve()
    if run(['git', '-C', fission, 'rev-parse', 'HEAD']).strip() != profile['fission_commit']:
        raise ValueError('Fission revision/profile mismatch')
    if run(['git', '-C', fission, 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Fission has tracked changes')
    if sha((fission/'utils/sleigh-specs/compiled/x86/x86-64.sla').read_bytes()) != profile['sla_sha256']:
        raise ValueError('SLA/profile mismatch')
    forge = args.forge.resolve()
    cli = fission/'target/release/fission_cli'
    output.mkdir(parents=True)
    for folder in ['sources', 'binaries', 'attempts']: (output/folder).mkdir()
    shutil.copyfile(args.config, output/'config.json')
    (output/'profile.json').write_bytes(profile_bytes)
    programs = [program(f, i, cfg['seed'], split) for f, split in sorted(cfg['families'].items()) for i in range(cfg['variants_per_family'] + cfg.get('extra_variants', {}).get(f, 0))]
    inputs = probes()
    for p in programs:
        p['oracle_outputs'] = [oracle(p, a, b) for a, b in inputs]
        p['behavior_sha256'] = sha(canonical(p['oracle_outputs']))
    families = {}
    for family in sorted(cfg['families']):
        family_programs = [p for p in programs if p['family'] == family]
        source = '_Static_assert(sizeof(unsigned int) == 4, "32-bit unsigned required");\n' + '\n'.join(p['source'].replace(' f(', ' '+p['compiled_name']+'(', 1) for p in family_programs) + '\n'
        path = output/'sources'/f'{family}.c'
        path.write_text(source)
        ast = json.loads(run([args.cc, '--target=x86_64-unknown-linux-gnu', '-std=c11', '-fsyntax-only', '-Xclang', '-ast-dump=json', path]))
        nodes = {n['name']:n for n in ast.get('inner', []) if n.get('kind') == 'FunctionDecl'}
        for p in family_programs:
            p['canonical_ast'] = canonical_ast(nodes[p['compiled_name']])
            p['shape_ast'] = canonical_ast(nodes[p['compiled_name']], True)
            p['ast_sha256'] = sha(canonical(p['canonical_ast']))
            p['shape_sha256'] = sha(canonical(p['shape_ast']))
        families[family] = (path, family_programs)
    (output/'programs.jsonl').write_text(''.join(json.dumps(p, separators=(',', ':'))+'\n' for p in programs))
    write(output/'probes.json', inputs)
    # Portable, generated C is executed only on the host as a source-label oracle.
    harness = ['#include <stdio.h>', '_Static_assert(sizeof(unsigned int)==4,"uint32");']
    harness += [f'unsigned int {p["compiled_name"]}(unsigned int,unsigned int);' for p in programs]
    harness.append('int main(void) {')
    for p in programs:
        for i, (a,b) in enumerate(inputs):
            harness.append(f'printf("{p["id"]} {i} %u\\n", {p["compiled_name"]}({a}u,{b}u));')
    harness.append('return 0; }')
    (output/'sources/native-check.c').write_text('\n'.join(harness)+'\n')
    native = output/'native-check'
    subprocess.run([args.cc, '-std=c11', '-O1', *map(str, [v[0] for v in families.values()]), str(output/'sources/native-check.c'), '-o', str(native)], check=True, timeout=180)
    observations = {}
    for line in run([native]).splitlines():
        identifier, index, value = line.split()
        key = (identifier, int(index))
        if key in observations: raise ValueError('duplicate native oracle observation')
        observations[key] = int(value)
    expected = {(p['id'], i):v for p in programs for i,v in enumerate(p['oracle_outputs'])}
    if observations != expected: raise ValueError('generated C / independent oracle mismatch')
    write(output/'native-verification.json', dict(status='passed', programs=len(programs), probes_per_program=len(inputs), comparisons=len(expected), scope='owned_host_C_vs_generator_oracle_only_not_x86_IR_parity'))
    print(json.dumps({'phase':'native_labels_checked', 'programs':len(programs), 'comparisons':len(expected)}), flush=True)
    env = os.environ.copy()
    env['FISSION_SLEIGH_SPEC_DIR'] = str(fission/'utils/sleigh-specs')
    env['FISSION_GHIDRA_DATA_DIR'] = str(fission/'utils/ghidra-data')
    attempts = []
    manifest = dict(schema_version=1, name=cfg['name'], origin='owned_synthetic_generator', split_unit='generator_family',
                    representation=cfg['representation'], config_sha256=sha((output/'config.json').read_bytes()),
                    profile_sha256=sha(profile_bytes), source_license='repository Apache-2.0',
                    compiler=run([args.cc, '--version']).splitlines()[0], fission_commit=profile['fission_commit'],
                    fission_cli_sha256=sha(cli.read_bytes()), forge_binary_sha256=sha(forge.read_bytes()),
                    forge_revision=run(['git','rev-parse','HEAD']).strip(), forge_worktree_dirty=bool(run(['git','status','--porcelain']).strip()),
                    python_version=run(['python3','--version']).strip(), generator_sha256=sha(Path(__file__).read_bytes()), expression_generator_sha256=sha(Path(expressions.__file__).read_bytes()),
                    compiler_flags=['--target=x86_64-unknown-linux-gnu','-std=c11','-g0','-fno-inline','-fno-builtin','-fno-stack-protector','-fno-vectorize','-fno-slp-vectorize','-fno-unroll-loops','-nostdlib','-fuse-ld=lld'],
                    external_evaluation_overlap='not_measured', benchmark_sources_used=False)
    if any(f in (shallow.TEMPLATES|shallow.EXTRA_TEMPLATES) for f in cfg['families']):manifest['shallow_generator_sha256']=sha(Path(shallow.__file__).read_bytes())
    write(output/'manifest.json', manifest)
    with (output/'pool.jsonl').open('w') as pool:
        for family, (path, items) in families.items():
            for optimization in cfg['optimizations']:
                relative = f'binaries/{family}-{optimization}.elf'
                binary_path = output/relative
                command = [args.cc, *manifest['compiler_flags'], '-'+optimization,
                           '-Wl,-e,'+items[0]['compiled_name'], '-Wl,--build-id=none', str(path), '-o', str(binary_path)]
                subprocess.run(command, check=True, timeout=180)
                binary = binary_path.read_bytes()
                symbols = {}
                for line in run([args.nm,'--defined-only','--print-size','--format=posix',binary_path]).splitlines():
                    fields = line.split()
                    if len(fields)>=4: symbols[fields[0]] = (int(fields[2],16),int(fields[3],16))
                for p in items:
                    identifier = p['id']+'-'+optimization
                    attempt = dict(id=identifier, program_id=p['id'], family=family, split=p['split'], optimization=optimization)
                    attempts.append(attempt)
                    folder = output/'attempts'/identifier
                    folder.mkdir()
                    try:
                        address, size = symbols[p['compiled_name']]
                        code = function_bytes(binary,address,size)
                        raw_text = run([cli,'--resource-root',fission/'utils','raw-pcode',binary_path,'--addr',f'0x{address:x}',
                                        '--language',profile['language'],'--max-bytes',size,'--no-db','--json'],env=env)
                        raw_path = folder/'raw.json'
                        raw_path.write_text(raw_text)
                        report = json.loads(run([forge,'pcode-compile','--raw',raw_path,'--profile',output/'profile.json',
                                                 '--representation',cfg['representation'],'--output',folder/'ir']))
                        pair = dict(schema_version=1,id=identifier,input=dict(representation=cfg['representation'],profile=profile,raw=json.loads(raw_text)),source=p['source'],
                                    provenance=dict(binary_sha256=sha(binary),raw_sha256=report['raw_canonical_sha256'],source_sha256=sha(p['source'].encode()),extraction='owned_synthetic_corpus_static_lift'))
                        metadata = dict(id=identifier,program_id=p['id'],family=family,split=p['split'],optimization=optimization,
                                        binary_path=relative,address=address,code_size=size,machine_code_sha256=sha(code),
                                        ast_sha256=p['ast_sha256'],shape_sha256=p['shape_sha256'],behavior_sha256=p['behavior_sha256'],
                                        prompt_sha256=report['prompt_sha256'],sequence_length=len(report['prompt'].encode())+len(p['source'].encode())+2)
                        pool.write(json.dumps(dict(pair=pair,metadata=metadata),separators=(',',':'))+'\n')
                        attempt.update(status='supported',prompt_bytes=len(report['prompt'].encode()),sequence_length=metadata['sequence_length'])
                    except (ValueError,KeyError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                        diagnostic = getattr(error,'stderr',None) or str(error)
                        if isinstance(diagnostic,bytes): diagnostic=diagnostic.decode(errors='replace')
                        attempt.update(status='unsupported',diagnostic=diagnostic[-4000:])
                print(json.dumps({'phase':'lifted','family':family,'optimization':optimization,'attempted':len(attempts),'supported':sum(a['status']=='supported' for a in attempts)}),flush=True)
    (output/'attempts.jsonl').write_text(''.join(json.dumps(a,separators=(',',':'))+'\n' for a in attempts))
    manifest['artifacts'] = {str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*'))
                             if p.is_file() and p.name!='manifest.json' and 'attempts/' not in str(p.relative_to(output))}
    write(output/'manifest.json',manifest)
    audit = json.loads(run([forge,'corpus-audit','--root',output,'--finalize']))
    print(json.dumps({k:audit[k] for k in ['attempted','supported','unsupported','splits','quarantined_families']},sort_keys=True),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/corpus/bootstrap-v1.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=Path('configs/lifter/fission-x86_64.json'))
    parser.add_argument('--forge',type=Path,default=Path('target/debug/forge'))
    parser.add_argument('--fission-root',type=Path,default=Path('external/Fission'))
    parser.add_argument('--cc',default='clang')
    parser.add_argument('--nm',default='llvm-nm')
    build(parser.parse_args())


if __name__=='__main__': main()
