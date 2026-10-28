"""Score raw C completions from any generator with the frozen functional protocol.

Uses the same compiler, bounded Wasm runtime, probes and `functional.assess` as
`score_gpu.py`. Output bytes are scored unchanged (no repair, prefix or format
conversion); only the 512-byte output cap of the protocol is enforced and
reported. Completions are keyed by validation ID; missing IDs count as failures.
"""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import functional as f


def load_completions(path):
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get('raw_base64') is not None:
            raw = base64.b64decode(r['raw_base64'])
        elif 'text' in r:
            raw = r['text'].encode()
        else:
            raw = None
        out[r['id']] = raw
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen', type=Path, required=True, help='functional.py freeze output (manifest, references, inputs)')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--completions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', default='external')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    a = p.parse_args()
    e = a.frozen.resolve()
    manifest = f.read(e / 'protocol-manifest.json' if (e / 'protocol-manifest.json').exists() else e / 'manifest.json')
    references = f.read(e / 'references.json')
    rows = {r['id']: r for r in map(json.loads, (a.corpus / 'release' / (a.split + '.jsonl')).read_text().splitlines())}
    completions = load_completions(a.completions)
    cap = manifest['settings']['max_output_bytes']
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    total = dict(targets=0, compiled=0, all_cases_match=0, matched_cases=0, executed_cases=0, targets_reaching_execution=0,
                 statuses=Counter(), truncated_to_cap=0, missing=0)
    exported = []
    reference_cases = 0
    for i, target in enumerate(manifest['targets']):
        identifier = target['id']
        folder = out / identifier
        folder.mkdir()
        row = rows[identifier]
        if f.sha(row['source'].encode()) != target['source_sha256'] or f.sha(f.canonical(references[identifier])) != target['oracle_sha256']:
            raise ValueError('original label changed')
        reference = f.compile_c(row['source'].encode(), folder, 'reference', manifest['compiler'])
        execution = f.run_module(folder / 'reference.wasm', e / 'inputs.json', folder, 'reference', manifest['settings'])
        if reference['status'] != 'compiled' or [c.get('value') for c in execution['cases']] != references[identifier]:
            raise ValueError('original reference no longer reproduces')
        reference_cases += len(execution['cases'])
        raw = completions.get(identifier)
        destination = folder / a.arm
        destination.mkdir()
        if raw is None:
            total['missing'] += 1
            result = dict(status='generation_failure', compiled=False, all_cases_match=False, matched_cases=0)
        else:
            if len(raw) > cap:
                total['truncated_to_cap'] += 1
                raw = raw[:cap]
            result = f.assess(raw, references[identifier], e / 'inputs.json', destination, manifest['compiler'], manifest['settings'])
        f.write(destination / 'result.json', result)
        total['targets'] += 1
        total['compiled'] += result['compiled']
        total['all_cases_match'] += result['all_cases_match']
        total['matched_cases'] += result['matched_cases']
        total['statuses'][result['status']] += 1
        if 'execution' in result:
            total['targets_reaching_execution'] += 1
            total['executed_cases'] += len(result['execution'].get('cases', []))
        exported.append(dict(id=identifier, status=result['status'], compiled=result['compiled'], all_cases_match=result['all_cases_match'],
                             matched_cases=result['matched_cases'], raw_base64=base64.b64encode(raw).decode() if raw is not None else None,
                             source_sha256=target['source_sha256']))
        if (i + 1) % 40 == 0:
            print('scored', i + 1, flush=True)
    n = len(manifest['targets'])
    total['compile_rate'] = total['compiled'] / n
    total['all_case_match_rate'] = total['all_cases_match'] / n
    total['case_match_rate'] = total['matched_cases'] / reference_cases
    (out / 'completions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in exported))
    report = dict(schema_version=1, status='complete', scope='owned_synthetic_%s_not_DecBench' % a.split, arm=a.arm,
                  reference_manifest_sha256=f.sha((e / ('protocol-manifest.json' if (e / 'protocol-manifest.json').exists() else 'manifest.json')).read_bytes()),
                  completions_sha256=f.sha(a.completions.read_bytes()), scorer_sha256=f.sha(Path(__file__).read_bytes()),
                  reference_comparisons=reference_cases, expected_targets=n, arms={a.arm: total}, test_model_examples=0,
                  raw_output_policy='unchanged_C_no_repairs_output_cap_only')
    f.write(out / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
