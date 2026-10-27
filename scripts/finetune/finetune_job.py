"""Colab GPU job: full fine-tune a pretrained causal LM on frozen prompt/C pairs.

Protocol mirrors the Forge corpus runs: completion-only loss, token-weighted
validation NLL, checkpoint chosen by validation NLL only (never by functional
score), greedy raw generation. The reserved test split is never loaded.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path('/content/ft')
RESULT = Path('/content/ft-results')
STATUS = Path('/content/ft-status.json')
MARKER = '### C\n'


def status(**kw):
    STATUS.write_text(json.dumps(kw))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    return [json.loads(l) for l in (ROOT / 'data' / name).read_text().splitlines()]


def main():
    job = json.loads((ROOT / 'job.json').read_text())
    RESULT.mkdir(exist_ok=True)
    status(state='running', phase='setup')
    try:
        import torch
        import transformers
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'accelerate'], check=True)
        import torch
        import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = job['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(job['model'], revision=job['revision'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(job['model'], revision=job['revision'], torch_dtype=torch.float32).cuda()
    params = sum(p.numel() for p in model.parameters())
    (RESULT / 'environment.json').write_text(json.dumps(dict(
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
        parameters=params, model=job['model'], revision=job['revision'], eos_token_id=eos,
        data={n: sha(ROOT / 'data' / n) for n in ['train.jsonl', 'validation.jsonl', 'train-probe.jsonl']}), indent=2))

    def encode(r):
        prompt = tok(r['prompt'] + MARKER, add_special_tokens=False)['input_ids']
        completion = tok(r['source'], add_special_tokens=False)['input_ids'] + [eos]
        return prompt, completion

    train, valid = rows('train.jsonl'), rows('validation.jsonl')
    train_ids = [encode(r) for r in train]
    valid_ids = [encode(r) for r in valid]
    longest = max(len(p) + len(c) for p, c in train_ids + valid_ids)
    status(state='running', phase='train', step=0, longest_sequence=longest)

    def collate(items):
        n = max(len(p) + len(c) for p, c in items)
        ids = torch.full((len(items), n), tok.pad_token_id)
        labels = torch.full((len(items), n), -100)
        mask = torch.zeros((len(items), n), dtype=torch.long)
        for i, (p, c) in enumerate(items):
            seq = p + c
            ids[i, :len(seq)] = torch.tensor(seq)
            labels[i, len(p):len(seq)] = torch.tensor(c)
            mask[i, :len(seq)] = 1
        return ids.cuda(), labels.cuda(), mask.cuda()

    @torch.no_grad()
    def validation_nll():
        model.eval()
        total, count = 0.0, 0
        for i in range(0, len(valid_ids), 16):
            ids, labels, mask = collate(valid_ids[i:i + 16])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction='sum')
            total += loss.item()
            count += (labels[:, 1:] != -100).sum().item()
        model.train()
        return total / count

    batch, epochs, lr = job['batch_size'], job['epochs'], job['learning_rate']
    steps_per_epoch = math.ceil(len(train_ids) / batch)
    total_steps = steps_per_epoch * epochs
    warmup = job['warmup_steps']
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    best = dict(nll=float('inf'), step=0)
    best_state = None
    log = (RESULT / 'metrics.jsonl').open('w')
    start = time.monotonic()
    model.train()
    step = 0
    rng = random.Random(seed)
    initial = validation_nll()
    log.write(json.dumps(dict(event='validation', step=0, nll=initial)) + '\n')
    for epoch in range(epochs):
        order = list(range(len(train_ids)))
        rng.shuffle(order)
        for i in range(0, len(order), batch):
            ids, labels, mask = collate([train_ids[j] for j in order[i:i + batch]])
            for group in opt.param_groups:
                group['lr'] = lr * scale(step)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps(dict(event='train', step=step, epoch=epoch, nll=loss.item())) + '\n')
            if step % job['evaluate_every'] == 0 or step == total_steps:
                nll = validation_nll()
                log.write(json.dumps(dict(event='validation', step=step, nll=nll)) + '\n')
                log.flush()
                if nll < best['nll']:
                    best = dict(nll=nll, step=step)
                    best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
                status(state='running', phase='train', step=step, total_steps=total_steps, best=best,
                       elapsed=time.monotonic() - start)
    log.close()
    train_seconds = time.monotonic() - start
    final_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}

    tok.padding_side = 'left'

    @torch.no_grad()
    def generate(items, name):
        model.eval()
        out = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            enc = tok([r['prompt'] + MARKER for r in chunk], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                gen = model.generate(**enc, do_sample=False, max_new_tokens=job['max_new_tokens'],
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for r, seq in zip(chunk, gen[:, enc['input_ids'].shape[1]:]):
                tokens = seq.tolist()
                hit_eos = eos in tokens
                if hit_eos:
                    tokens = tokens[:tokens.index(eos)]
                text = tok.decode(tokens, skip_special_tokens=False)
                out.append(dict(id=r['id'], text=text, ended_with_eos=hit_eos, tokens=len(tokens)))
        (RESULT / f'{name}.jsonl').write_text(''.join(json.dumps(o) + '\n' for o in out))

    probe = rows('train-probe.jsonl')
    for label, state in [('selected', best_state), ('final', final_state)]:
        status(state='running', phase='generate-' + label, step=total_steps)
        model.load_state_dict(state)
        generate(valid, f'validation-{label}')
        if label == 'selected':
            generate(probe, 'train-probe-selected')
    (RESULT / 'training-report.json').write_text(json.dumps(dict(
        job=job, parameters=params, steps_per_epoch=steps_per_epoch, total_steps=total_steps,
        initial_validation_nll=initial, best_validation=best, final_step=total_steps,
        train_wall_seconds=train_seconds, longest_sequence=longest, test_model_examples=0), indent=2))
    subprocess.run(['tar', '-czf', '/content/ft-results.tar.gz', '-C', '/content', 'ft-results'], check=True)
    status(state='complete', phase='done', best=best, train_wall_seconds=train_seconds)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise
