"""Derive an evaluation plan and model config for a new corpus from a previous plan.

Only the corpus lock, tokenizer, update count and names change; every protocol setting is copied from the base
plan. The vocabulary is 256 bytes + merges + 4 special tokens (checked against the base plan).
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-plan', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True, help='tokenizer.json from tokenizer-train')
    p.add_argument('--name', required=True)
    p.add_argument('--scope', required=True)
    p.add_argument('--kind', required=True)
    p.add_argument('--curriculum-policy', required=True)
    p.add_argument('--epochs', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    a = p.parse_args()
    base = json.loads(a.base_plan.read_text())
    tokenizer = json.loads(a.tokenizer.read_text())
    base_tokenizer = base['tokenizer']
    if 256 + len(base_tokenizer['merges']) + 4 != base['model']['vocab_size']:
        raise SystemExit('vocabulary rule does not reproduce the base plan')
    release = a.corpus / 'release'
    rows = len((release / 'train.jsonl').read_text().splitlines())
    steps = -(-rows * a.epochs // a.batch_size)
    plan = dict(base)
    plan.update(name=a.name, scope=a.scope, corpus_lock_sha256=__import__('hashlib').sha256((release / 'lock.json').read_bytes()).hexdigest(),
                tokenizer=tokenizer, curriculum_policy=a.curriculum_policy)
    plan['model'] = dict(base['model'], vocab_size=256 + len(tokenizer['merges']) + 4)
    plan['improvement'] = dict(base['improvement'], kind=a.kind, additional_updates=steps, final_step=steps, batch_size=a.batch_size)
    a.output.write_text(json.dumps(plan, indent=2) + '\n')
    a.model_output.write_text(json.dumps(plan['model'], indent=2) + '\n')
    print(json.dumps(dict(train_rows=rows, steps=steps, vocab_size=plan['model']['vocab_size'], plan=str(a.output))))


if __name__ == '__main__':
    main()
