"""Static DecBench eval-kit adapter; Rust handles IR and model inference.

Writes a separate run directory; never executes kit binaries, generated C, or
kit-supplied scripts. No scoring, repair, upload, or submission is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def manifest_targets(data):
    value = json.loads(data, object_pairs_hook=unique_object)
    if (not isinstance(value, dict) or type(value.get('kit_format_version')) is not int
            or value['kit_format_version'] != 1 or not isinstance(value.get('public'), dict)):
        raise ValueError('expected DecBench kit format 1 with public target mapping')
    targets = []
    for binary, addresses in sorted(value['public'].items()):
        if not isinstance(binary, str) or Path(binary).name != binary or '\\' in binary or binary in ('', '.', '..'):
            raise ValueError('binary names must be plain filenames')
        if not isinstance(addresses, list):
            raise ValueError('target addresses must be a list')
        seen = set()
        for address in addresses:
            if isinstance(address, bool) or not isinstance(address, (int, str)):
                raise ValueError('invalid address type')
            if isinstance(address, str) and not re.fullmatch(r'(?:0x)?[0-9a-fA-F]+', address):
                raise ValueError('invalid hexadecimal address')
            number = address if isinstance(address, int) else int(address, 16)
            if not 0 <= number < 2**64 or number in seen:
                raise ValueError('out-of-range or duplicate target address')
            seen.add(number)
            targets.append((binary, f'0x{number:x}'))
    if not targets:
        raise ValueError('empty target manifest')
    return targets


def command(args, **kwargs):
    return subprocess.check_output([str(a) for a in args], text=True, stderr=subprocess.PIPE, timeout=120, **kwargs)


def function_name(text, clang):
    # The first adapter accepts plain standalone C and delegates parsing to Clang.
    # No repair or reference-source includes are injected.
    if any(line.lstrip().startswith('#') for line in text.splitlines()) or '\x00' in text:
        raise ValueError('preprocessor directives/NUL are outside the initial C-output contract')
    tree = json.loads(command([clang, '-x', 'c', '-fsyntax-only', '-fno-color-diagnostics',
                               '-Xclang', '-ast-dump=json', '-'], input=text))
    names = [node['name'] for node in tree.get('inner', []) if node.get('kind') == 'FunctionDecl'
             and any(child.get('kind') == 'CompoundStmt' for child in node.get('inner', []))]
    if len(names) != 1:
        raise ValueError('expected exactly one generated top-level C function definition')
    return names[0]


def run(args):
    kit = args.kit.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'output already exists: {output}')
    manifest_bytes = (kit / 'functions.json').read_bytes()
    targets = manifest_targets(manifest_bytes)  # The private mapping is not used.
    fission = args.fission_root.resolve()
    forge = args.forge.resolve()
    profile_path = args.profile.resolve()
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    actual_revision = command(['git', '-C', fission, 'rev-parse', 'HEAD']).strip()
    if actual_revision != profile['fission_commit']:
        raise ValueError('Fission checkout revision differs from profile')
    if command(['git', '-C', fission, 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Fission checkout has tracked modifications')
    sla = fission / 'utils/sleigh-specs/compiled/x86/x86-64.sla'
    if sha(sla.read_bytes()) != profile['sla_sha256']:
        raise ValueError('SLA digest differs from profile')
    cli = fission / 'target/release/fission_cli'
    state = None
    if not args.prepare_only:
        if args.checkpoint is None:
            raise ValueError('--checkpoint is required for generation')
        args.checkpoint = args.checkpoint.resolve()
        state = json.loads((args.checkpoint / 'state.json').read_text())
        if state.get('schema_version') != 3 or args.representation not in state.get('representations', []):
            raise ValueError('benchmark adapter requires checkpoint schema 3 with a matching input lane')
    output.mkdir(parents=True)
    (output / 'functions.json').write_bytes(manifest_bytes)
    (output / 'results').mkdir()
    env = os.environ.copy()
    env['FISSION_SLEIGH_SPEC_DIR'] = str(fission / 'utils/sleigh-specs')
    env['FISSION_GHIDRA_DATA_DIR'] = str(fission / 'utils/ghidra-data')
    report = dict(schema_version=1, status='running', scope='external_kit_adapter_only_no_scores',
                  selection='all public targets; ELF64 little-endian x86-64 headers only; no source-language oracle',
                  forge_revision=command(['git', '-C', Path(__file__).resolve().parents[2], 'rev-parse', 'HEAD']).strip(),
                  forge_worktree_dirty=bool(command(['git', '-C', Path(__file__).resolve().parents[2], 'status', '--porcelain']).strip()),
                  forge_binary_sha256=sha(forge.read_bytes()), fission_commit=actual_revision,
                  fission_cli_sha256=sha(cli.read_bytes()), profile_sha256=sha(profile_bytes),
                  functions_json_sha256=sha(manifest_bytes), representation=args.representation,
                  max_new_tokens=args.max_new_tokens, prepare_only=args.prepare_only,
                  max_targets=args.max_targets, manifest_targets=len(targets), device=args.device, targets=[])
    if state is not None:
        report.update(checkpoint_sha256=sha((args.checkpoint / 'training.safetensors').read_bytes()),
                      checkpoint_state_sha256=sha((args.checkpoint / 'state.json').read_bytes()))
    submission = dict(decompiler=dict(name='Fission Forge', version=report['forge_revision'] + ('+dirty' if report['forge_worktree_dirty'] else '')), results={})
    generated = {}
    for index, (binary_name, address) in enumerate(targets):
        record = dict(binary=binary_name, address=address)
        report['targets'].append(record)
        phase = 'selection'
        try:
            if args.max_targets is not None and index >= args.max_targets:
                record.update(status='excluded_budget')
                continue
            binary = (kit / 'binaries' / binary_name).resolve()
            if binary.parent != (kit / 'binaries').resolve():
                raise ValueError('binary path escapes kit binaries directory')
            data = binary.read_bytes()
            record['binary_sha256'] = sha(data)
            if not (len(data) >= 20 and data[:6] == b'\x7fELF\x02\x01' and int.from_bytes(data[18:20], 'little') == 62):
                record.update(status='unsupported_binary_format')
                continue
            attempt = output / f'target_{index:05d}'
            attempt.mkdir()
            phase = 'lifting'
            record['lift_attempted'] = True
            raw = command([cli, '--resource-root', fission / 'utils', 'raw-pcode', binary,
                           '--addr', address, '--language', profile['language'], '--no-db', '--json'], env=env)
            raw_path = attempt / 'raw.json'
            raw_path.write_text(raw)
            record['raw_sha256'] = sha(raw.encode())
            phase = 'ir_validation'
            compiled = json.loads(command([forge, 'pcode-compile', '--raw', raw_path, '--profile', profile_path,
                                           '--representation', args.representation, '--output', attempt / 'ir']))
            record.update(prompt_sha256=compiled['prompt_sha256'], prompt_bytes=len(compiled['prompt'].encode()),
                          blocks=compiled['blocks'], phis=compiled['phis'], opaque_calls=compiled['opaque_calls'])
            if args.prepare_only:
                record.update(status='prepared_only')
                continue
            phase = 'generation'
            record['generation_attempted'] = True
            completion = json.loads(command([forge, 'generate', '--checkpoint', args.checkpoint,
                    '--prompt', attempt / 'ir/prompt.txt', '--representation', args.representation,
                    '--max-new-tokens', args.max_new_tokens, '--device', args.device, '--output', attempt / 'generation']))
            record['generation_stop_reason'] = completion['completion']['stop_reason']
            if record['generation_stop_reason'] != 'eos':
                record.update(status='incomplete_generation')
                continue
            if completion['text'] is None:
                record.update(status='non_utf8_generation')
                continue
            phase = 'c_validation'
            text = completion['text']
            name = function_name(text, args.clang)
            functions = generated.setdefault(binary_name, {})
            if name in functions:
                raise ValueError('generated function identifier collides within this binary')
            # Check the combined translation unit too: separate valid outputs may
            # still conflict through typedefs, globals, or declarations.
            combined = '\n\n'.join([entry[1] for entry in functions.values()] + [text]) + '\n'
            command([args.clang, '-x', 'c', '-fsyntax-only', '-fno-color-diagnostics', '-'], input=combined)
            functions[name] = (address, text)
            record.update(status='emitted', function_identifier=name, raw_output_sha256=sha(text.encode()))
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            diagnostic = getattr(error, 'stderr', None) or str(error)
            if isinstance(diagnostic, bytes):
                diagnostic = diagnostic.decode(errors='replace')
            record.update(status='failed', phase=phase, diagnostic=diagnostic[-8000:])
        finally:
            (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    for binary, functions in generated.items():
        if not functions:
            continue
        filename = f'binary_{len(submission["results"]):05d}.c'
        (output / 'results' / filename).write_text('\n\n'.join(text for _, text in functions.values()) + '\n')
        submission['results'][filename] = dict(binary=binary, functions={name:address for name,(address,_) in functions.items()})
    (output / 'results/results.json').write_text(json.dumps(submission, indent=2) + '\n')
    report['status'] = 'complete'
    counts = {}
    for target in report['targets']:
        counts[target['status']] = counts.get(target['status'], 0) + 1
    report['outcomes'] = counts
    report['attempted_lifts'] = sum(bool(t.get('lift_attempted')) for t in report['targets'])
    report['attempted_generations'] = sum(bool(t.get('generation_attempted')) for t in report['targets'])
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='complete', outcomes=counts, scoring='not_run'), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--max-targets', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--fission-root', type=Path, default=Path('external/Fission'))
    parser.add_argument('--forge', type=Path, default=Path('target/debug/forge'))
    parser.add_argument('--profile', type=Path, default=Path('configs/lifter/fission-x86_64.json'))
    parser.add_argument('--representation', choices=['pcode-plus-ssa-v0','pcode-plus-ssa-v1'], default='pcode-plus-ssa-v1')
    parser.add_argument('--clang', default='clang')
    parser.add_argument('--device', choices=['cpu','cuda','metal'], default='cpu')
    args = parser.parse_args()
    if args.max_targets is not None and args.max_targets <= 0:
        parser.error('--max-targets must be positive')
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error('--max-new-tokens must be 1..4096')
    run(args)


if __name__ == '__main__':
    main()
