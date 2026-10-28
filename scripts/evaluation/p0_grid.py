"""Run the predeclared P0/P1 grid locally on CPU and summarize copy learning.

P0 corpus: one template (`a + k`), disjoint literals across splits (diagnostic mode).
For each run: train with `forge train-corpus`, select the checkpoint with the lowest
validation NLL, regenerate validation rows (held-out literals) and a stride of training
rows, and score validation functions with the frozen protocol. Copy is "learned" when at
least 90% of held-out literals are right. Diagnostic only; test rows are never used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

RUNS = [
    dict(name='r1-b1-1440-lr3e-4', batch=1, steps=1440, lr=0.0003),
    dict(name='r2-b1-7200-lr3e-4', batch=1, steps=7200, lr=0.0003),
    dict(name='r3-b8-1440-lr3e-4', batch=8, steps=1440, lr=0.0003),
    dict(name='r4-b8-1440-lr1e-3', batch=8, steps=1440, lr=0.001),
    dict(name='r5-b1-1440-lr1e-3', batch=1, steps=1440, lr=0.001),
    dict(name='r6-b8-180-lr3e-4', batch=8, steps=180, lr=0.0003),
    dict(name='r7-b8-360-lr3e-4', batch=8, steps=360, lr=0.0003),
]


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--forge', type=Path, default=Path('target/release/forge'))
    p.add_argument('--python', default='outputs/eval-venv/bin/python')
    p.add_argument('--only', nargs='*', help='run names to execute')
    p.add_argument('--skip-score', action='store_true', help='skip the frozen functional scoring (needs the evaluation venv)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    a.output = a.output.resolve()
    summary = {}
    for spec in RUNS:
        if a.only and spec['name'] not in a.only:
            continue
        out = a.output / spec['name']
        every = max(1, spec['steps'] // 12)
        run([a.forge, 'train-corpus', '--config', a.model, '--corpus', a.corpus, '--tokenizer', a.tokenizer, '--output', out / 'train',
             '--steps', spec['steps'], '--batch-size', spec['batch'], '--learning-rate', spec['lr'], '--evaluate-every', every,
             '--checkpoint-every', every, '--seed', 42, '--device', 'cpu'], stdout=subprocess.DEVNULL)
        rows = [json.loads(l) for l in (out / 'train/metrics.jsonl').read_text().splitlines()]
        evals = [(r['validation']['step'], r['validation']['nll']) for r in rows if r.get('event') == 'validation']
        best_step = min(evals, key=lambda e: e[1])[0]
        result = dict(spec=spec, validation_nll_by_step=[[s, round(n, 4)] for s, n in evals], selected_step=best_step, arms={})
        for arm, step in [('selected', best_step), ('final', spec['steps'])]:
            checkpoint = out / 'train/checkpoints' / f'step-{step:08d}'
            arm_result = {}
            for split, stride in [('validation', 1), ('train', 6)]:
                target = out / f'{arm}-{split}'
                cmd = [sys.executable, 'scripts/evaluation/constant_reproduction.py', '--corpus', a.corpus, '--checkpoint', checkpoint,
                       '--split', split, '--stride', stride, '--output', target]
                if split == 'validation':
                    cmd += ['--completions-out', out / f'{arm}-validation-completions.jsonl']
                run(cmd, stdout=subprocess.DEVNULL)
                s = json.loads((target / 'summary.json').read_text())
                arm_result[split] = dict(rows=s['rows'], literals_match=s['literals_match'], skeleton_match=s['skeleton_match'],
                                         exact=s['exact'], distinct=s['distinct_generated_expressions'],
                                         top_literals=s['most_common_generated_literals'][:3])
            if a.skip_score:
                result['arms'][arm] = arm_result
                continue
            score_dir = out / f'{arm}-score'
            run([root / a.python, 'score_external.py', '--frozen', (root / a.frozen), '--corpus', (root / a.corpus),
                 '--completions', (out / f'{arm}-validation-completions.jsonl').resolve(), '--output', score_dir.resolve(), '--arm', arm],
                cwd=root / 'scripts/evaluation', stdout=subprocess.DEVNULL)
            report = json.loads((score_dir / 'report.json').read_text())['arms'][arm]
            arm_result['functional'] = dict(compiled=report['compiled'], matched=report['all_cases_match'], cases=report['matched_cases'])
            result['arms'][arm] = arm_result
        summary[spec['name']] = result
        (a.output / f"{spec['name']}.summary.json").write_text(json.dumps(result, indent=2) + '\n')
        v = result['arms']['selected']['validation']
        print(spec['name'], 'selected step', best_step, 'held-out literals', v['literals_match'], '/', v['rows'],
              '| final', result['arms']['final']['validation']['literals_match'], flush=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
