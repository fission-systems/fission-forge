"""Fine-tune a pretrained LM on one named Colab GPU session, then always stop it.

Credentials stay in the local Colab CLI; no interactive auth is invoked.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import time


def checked(cmd, timeout=120, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, timeout=timeout, **kw)


def remote(session, code_path, timeout=60):
    return subprocess.run(['colab', 'exec', '-s', session, '-f', str(code_path)], capture_output=True, text=True, timeout=timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--gpu', choices=['A100', 'L4', 'T4', 'G4'], default='L4')
    p.add_argument('--model', default='Qwen/Qwen2.5-Coder-0.5B')
    p.add_argument('--revision', required=True)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--evaluate-every', type=int, default=25)
    p.add_argument('--budget-seconds', type=int, default=2400)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    job = dict(model=a.model, revision=a.revision, seed=42, learning_rate=a.learning_rate, epochs=a.epochs,
               batch_size=a.batch_size, warmup_steps=20, evaluate_every=a.evaluate_every, max_new_tokens=128)
    (out / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    archive = out / 'payload.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out / 'job.json', 'ft/job.json')
        tar.add(root / 'finetune_job.py', 'ft/finetune_job.py')
        for name in ['train.jsonl', 'validation.jsonl', 'train-probe.jsonl', 'manifest.json']:
            tar.add(a.data / name, f'ft/data/{name}')
    (out / 'setup.py').write_text("import tarfile\ntarfile.open('/content/ft-payload.tar.gz').extractall('/content')\nprint('extracted')\n")
    (out / 'launch.py').write_text(
        "import subprocess,sys,json,pathlib\n"
        "if pathlib.Path('/content/ft-status.json').exists() and json.loads(pathlib.Path('/content/ft-status.json').read_text()).get('state') in ('running','complete'):\n"
        "    print('already started')\n"
        "else:\n"
        "    log=open('/content/ft-driver.log','w')\n"
        "    p=subprocess.Popen([sys.executable,'/content/ft/finetune_job.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
        "    print({'pid':p.pid})\n")
    (out / 'poll.py').write_text(
        "import pathlib\n"
        "s=pathlib.Path('/content/ft-status.json')\n"
        "print(s.read_text() if s.exists() else '{\"state\":\"no-status\"}')\n"
        "l=pathlib.Path('/content/ft-driver.log')\n"
        "print('LOG', l.read_text()[-600:].replace(chr(10),' | ') if l.exists() else '')\n")
    started = time.monotonic()
    allocated = False
    try:
        checked(['colab', 'new', '-s', a.session, '--gpu', a.gpu], timeout=240)
        allocated = True
        checked(['colab', 'upload', '-s', a.session, archive, '/content/ft-payload.tar.gz'], timeout=180)
        for attempt in range(5):
            try:
                done = remote(a.session, out / 'setup.py')
                if done.returncode == 0 and 'extracted' in done.stdout:
                    break
            except subprocess.TimeoutExpired:
                pass
            if attempt == 4:
                raise RuntimeError('payload did not extract on the session')
            time.sleep(6)
        for attempt in range(4):
            try:
                launched = remote(a.session, out / 'launch.py')
                if launched.returncode == 0:
                    print(launched.stdout.strip(), flush=True)
                    break
            except subprocess.TimeoutExpired:
                pass
            if attempt == 3:
                raise RuntimeError('job did not launch')
            time.sleep(6)
        last = ''
        while True:
            if time.monotonic() - started > a.budget_seconds:
                raise TimeoutError('fine-tuning exceeded the wall-clock budget')
            try:
                polled = remote(a.session, out / 'poll.py', timeout=50)
            except subprocess.TimeoutExpired:
                print('Status observation timed out; retrying', flush=True)
                time.sleep(10)
                continue
            line = next((l for l in polled.stdout.splitlines() if l.startswith('{')), '{}')
            if line != last:
                print(line[:300], flush=True)
                last = line
            state = json.loads(line).get('state')
            if state == 'complete':
                break
            if state == 'failed':
                print(polled.stdout[-1500:], flush=True)
                raise RuntimeError('remote job failed')
            time.sleep(20)
        checked(['colab', 'download', '-s', a.session, '/content/ft-results.tar.gz', out / 'results.tar.gz'], timeout=300)
        checked(['colab', 'download', '-s', a.session, '/content/ft-status.json', out / 'status.json'], timeout=90)
        with tarfile.open(out / 'results.tar.gz') as tar:
            tar.extractall(out, filter='data')
    finally:
        if allocated:
            subprocess.run(['colab', 'download', '-s', a.session, '/content/ft-driver.log', str(out / 'driver.log')], capture_output=True, timeout=90)
            checked(['colab', 'stop', '-s', a.session], timeout=120)


if __name__ == '__main__':
    main()
