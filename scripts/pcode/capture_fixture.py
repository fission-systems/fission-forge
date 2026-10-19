"""Compile owned C fixtures, lift statically with Fission, and prepare IR pairs.

No target binary is executed. Supported and unsupported attempts are both recorded.
Run from the Forge repository root; see docs/pcode.md for prerequisites.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, stderr=subprocess.PIPE, timeout=180, **kwargs)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fission-root', type=Path, default=Path('external/Fission'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forge', type=Path, default=Path('target/debug/forge'))
    parser.add_argument('--fixture-dir', type=Path, default=Path('fixtures/pcode'))
    parser.add_argument('--entry', default='forge_add')
    parser.add_argument('--representation', choices=['pcode-plus-ssa-v0','pcode-plus-ssa-v1'], default='pcode-plus-ssa-v0')
    parser.add_argument('--cc', default='clang')
    parser.add_argument('--nm', default='llvm-nm')
    parser.add_argument('--rustc', default='rustc', help='Must match the Fission build compiler')
    args = parser.parse_args()
    root = Path.cwd()
    fission = args.fission_root.resolve()
    forge = args.forge.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    if run(['git', '-C', fission, 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Fission checkout has tracked modifications')
    revision = run(['git', '-C', fission, 'rev-parse', 'HEAD']).strip()
    cli = fission / 'target/release/fission_cli'
    sla = fission / 'utils/sleigh-specs/compiled/x86/x86-64.sla'
    libraries = list((fission / 'target/release/deps').glob('libfission_sleigh-*.rlib'))
    if len(libraries) != 1:
        raise ValueError('Expected one built fission-sleigh rlib; use a clean dedicated target directory')
    output.mkdir(parents=True)
    helper = output / 'inspect-sla'
    subprocess.run([args.rustc, '--edition=2024', '-C', 'opt-level=2', '-C', 'lto=thin',
                    str(root / 'scripts/pcode/inspect_sla.rs'), '-L', f'dependency={fission / "target/release/deps"}',
                    '--extern', f'fission_sleigh={libraries[0]}', '-o', str(helper)], check=True, timeout=180)
    space_text = run([helper, sla])
    (output / 'spaces.txt').write_text(space_text)
    spaces = {row.split()[0]:tuple(map(int,row.split()[1:])) for row in space_text.splitlines()}
    assert spaces['ram'][1:] == (1, 8), 'requires byte-addressed 64-bit RAM'
    profile = dict(schema_version=1, language='x86:LE:64:default', endianness='little',
                   pointer_bytes=8, word_size=1, constant_space=spaces['const'][0],
                   unique_space=spaces['unique'][0], register_space=spaces['register'][0],
                   ram_space=spaces['ram'][0], fission_commit=revision, sla_sha256=digest(sla.read_bytes()))
    write(output / 'profile.json', profile)
    fixture_source = args.fixture_dir.resolve() / 'functions.c'
    labels = json.loads((args.fixture_dir.resolve() / 'labels.json').read_text())
    shutil.copyfile(fixture_source, output / 'functions.c')
    binary = output / 'functions.elf'
    command = [args.cc, '--target=x86_64-unknown-linux-gnu', '-O1', '-fno-inline', '-nostdlib',
               '-fuse-ld=lld', f'-Wl,-e,{args.entry}', '-Wl,--build-id=none', str(output / 'functions.c'), '-o', str(binary)]
    subprocess.run(command, check=True, timeout=180)
    symbols = {}
    for line in run([args.nm, '--defined-only', '--format=posix', binary]).splitlines():
        fields = line.split()
        if len(fields) >= 3:
            symbols[fields[0]] = '0x' + fields[2]
    lift_env = os.environ.copy()
    lift_env['FISSION_SLEIGH_SPEC_DIR'] = str(fission / 'utils/sleigh-specs')
    lift_env['FISSION_GHIDRA_DATA_DIR'] = str(fission / 'utils/ghidra-data')
    attempts, pairs = [], []
    for name, source in labels.items():
        record = dict(id=name, address=symbols.get(name))
        attempts.append(record)
        try:
            if name not in symbols:
                raise ValueError('function symbol missing')
            raw_text = run([cli, '--resource-root', fission / 'utils', 'raw-pcode', binary,
                            '--addr', symbols[name], '--language', profile['language'], '--no-db', '--json'], env=lift_env)
            raw_path = output / f'{name}.raw.json'
            raw_path.write_text(raw_text)
            raw = json.loads(raw_text)
            compiled = run([forge, 'pcode-compile', '--representation', args.representation, '--raw', raw_path, '--profile', output / 'profile.json',
                            '--output', output / f'{name}.ssa'])
            summary = json.loads(compiled)
            record.update(status='supported', operations=summary['operations'], prompt_sha256=summary['prompt_sha256'],
                          prompt_bytes=len(summary['prompt'].encode()), raw_file_sha256=digest(raw_text.encode()))
            pairs.append(dict(schema_version=1, id=name,
                              input=dict(representation=args.representation, profile=profile, raw=raw), source=source,
                              provenance=dict(binary_sha256=digest(binary.read_bytes()), raw_sha256=summary['raw_canonical_sha256'],
                                              source_sha256=digest(source.encode()), extraction='owned_c_fixture_static_lift')))
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            record.update(status='unsupported', reason=str(error), stderr=getattr(error, 'stderr', None))
    (output / 'pairs.jsonl').write_text(''.join(json.dumps(pair, separators=(',', ':')) + '\n' for pair in pairs))
    write(output / 'capture.json', dict(schema_version=1, evidence_scope='owned_fixture_extraction_only',
          representation=args.representation, fission_commit=revision, fission_cli_sha256=digest(cli.read_bytes()),
          compiler=run([args.cc, '--version']).splitlines()[0], compiler_arguments=command[1:],
          fixture_sha256=digest(fixture_source.read_bytes()), binary_sha256=digest(binary.read_bytes()),
          profile_sha256=digest((output / 'profile.json').read_bytes()),
          attempted=len(attempts), supported=len(pairs), attempts=attempts))
    print(json.dumps({'attempted':len(attempts), 'supported':len(pairs), 'output':str(output)}, indent=2))
    if not pairs:
        raise RuntimeError('No supported fixture; inspect capture.json')


if __name__ == '__main__':
    main()
