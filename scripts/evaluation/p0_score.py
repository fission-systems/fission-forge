"""Score the saved completions of a p0_grid.py run directory with the frozen protocol.

Never deletes anything: it refuses to overwrite an existing score directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    rows = {}
    for run in sorted(d for d in a.grid.iterdir() if d.is_dir()):
        summary_path = a.grid / f'{run.name}.summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for arm in ('selected', 'final'):
            target = run / f'{arm}-score'
            if not target.exists():
                subprocess.run([str(a.python), 'score_external.py', '--frozen', str(a.frozen.resolve()), '--corpus', str(a.corpus.resolve()),
                                '--completions', str((run / f'{arm}-validation-completions.jsonl').resolve()), '--output', str(target.resolve()),
                                '--arm', arm], cwd=here, check=True, stdout=subprocess.DEVNULL)
            report = json.loads((target / 'report.json').read_text())['arms'][arm]
            summary['arms'][arm]['functional'] = dict(compiled=report['compiled'], matched=report['all_cases_match'], cases=report['matched_cases'])
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        rows[run.name] = {arm: summary['arms'][arm]['functional']['matched'] for arm in ('selected', 'final')}
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
