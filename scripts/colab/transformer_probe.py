"""Run inside Colab after uploading a source-only archive to /content.

This is orchestration only; all model/autodiff/training code is Rust.
Outputs contain measurements, not a trained decompiler quality claim.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback
import urllib.request

ROOT = Path('/content/fission-forge')
RESULT = Path('/content/forge-probe-results')
STATUS = Path('/content/forge-probe-status.json')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, timeout=1800, **kwargs)


def measure(command, name, environment):
    """Sample whole-device memory, including baseline; not an allocator peak."""
    started = time.monotonic()
    peak_mib = 0
    with (RESULT / f'{name}.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started > 600:
                    raise TimeoutError(f'{name} exceeded 600 seconds')
                reading = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                peak_mib = max(peak_mib, int(reading.strip().splitlines()[0]))
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f'{name} exited {process.returncode}: {(RESULT / f"{name}.log").read_text()[-4000:]}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    write_json(RESULT / f'{name}-memory.json', {'observed_device_memory_peak_mib': peak_mib, 'sampling_interval_seconds': 0.5, 'includes_device_baseline': True, 'wall_seconds': time.monotonic()-started})
    print((RESULT / f'{name}.log').read_text(), flush=True)


def main():
    RESULT.mkdir()
    write_json(STATUS, {'state': 'running', 'phase': 'bootstrap'})
    archive = Path('/content/forge-source.tar.gz')
    write_json(RESULT / 'source.json', {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    ROOT.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(ROOT, filter='data')
    gpu = subprocess.check_output(['nvidia-smi'], text=True)
    print(gpu, flush=True)
    nvcc = subprocess.check_output(['nvcc', '--version'], text=True)
    environment = os.environ.copy()
    environment['PATH'] = str(Path.home() / '.cargo/bin') + ':' + environment['PATH']
    environment['CUDA_COMPUTE_CAP'] = '80'
    environment['CARGO_BUILD_JOBS'] = '4'
    if not shutil.which('cargo', path=environment['PATH']):
        installer = Path('/content/forge-rustup.sh')
        urllib.request.urlretrieve('https://sh.rustup.rs', installer)
        run(['sh', installer, '-y', '--profile', 'minimal', '--default-toolchain', 'stable'], env=environment)
    rust = subprocess.check_output(['rustc', '--version'], env=environment, text=True)
    write_json(RESULT / 'environment.json', {'gpu': gpu, 'nvcc': nvcc, 'rust': rust, 'cuda_compute_cap': '80', 'candle': '0.11.0'})
    write_json(STATUS, {'state': 'running', 'phase': 'cargo-build-cuda'})
    run(['cargo', 'build', '--locked', '--release', '-p', 'fission-forge', '--features', 'cuda'], cwd=ROOT, env=environment)
    binary = str(ROOT / 'target/release/forge')
    for name, config, steps, lr in [('tiny-cuda', 'tiny.json', '32', '0.003'), ('100m-cuda', '100m-byte-experimental.json', '2', '0.0001')]:
        write_json(STATUS, {'state': 'running', 'phase': name})
        measure([binary, 'train-smoke', '--config', f'configs/model/{config}', '--dataset', 'fixtures/smoke/pairs.jsonl', '--output', f'outputs/{name}', '--device', 'cuda', '--steps', steps, '--learning-rate', lr], name, environment)
        shutil.copy(ROOT / f'outputs/{name}/report.json', RESULT / f'{name}-report.json')
        shutil.copy(ROOT / f'outputs/{name}/state.json', RESULT / f'{name}-state.json')
    # Do not download gigabytes of synthetic model weights as if they were useful models.
    with tarfile.open('/content/forge-probe-results.tar.gz', 'w:gz') as archive:
        archive.add(RESULT, arcname='results')
    write_json(STATUS, {'state': 'complete', 'phase': 'done'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        write_json(STATUS, {'state': 'failed', 'error': traceback.format_exc()})
        raise
