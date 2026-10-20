"""Bounded Colab CUDA corpus pilot. Model/training remain Rust/Candle.

Uploaded by run_corpus.py; executes in /content. Archive is source + audited owned
corpus, never credentials. Always packages partial artifacts, including failures.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import traceback
import urllib.request

ROOT=Path('/content/forge-corpus')
RESULT=Path('/content/forge-corpus-results')
STATUS=Path('/content/forge-corpus-status.json')
ARCHIVE=Path('/content/forge-corpus-results.tar.gz')
START=time.monotonic()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def status(phase,state='running',**extra):write(STATUS,dict(state=state,phase=phase,elapsed_seconds=time.monotonic()-START,**extra))
def run(cmd,env=None,timeout=1200,**kwargs):
    print('+',' '.join(map(str,cmd)),flush=True)
    return subprocess.run(list(map(str,cmd)),env=env,check=True,timeout=timeout,**kwargs)


def measure(cmd,name,env,timeout):
    """Whole-device sampled memory; not allocator/transient peak, excludes no baseline."""
    status(name)
    started=time.monotonic();peak=0;sample_count=0
    with (RESULT/(name+'.log')).open('w') as log:
        p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic()-started>timeout or time.monotonic()-START>1500:raise TimeoutError(name+' exceeded phase/global wall budget')
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
                peak=max(peak,int(memory.strip().splitlines()[0]));sample_count+=1
                time.sleep(0.2)
            if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}): '+(RESULT/(name+'.log')).read_text()[-6000:])
        finally:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
            write(RESULT/(name+'-resources.json'),dict(wall_seconds=time.monotonic()-started,observed_whole_device_peak_mib=peak,sampling_interval_seconds=0.2,samples=sample_count,exit_code=p.returncode,includes_device_baseline=True))
    return json.loads(Path(cmd[list(cmd).index('--output')+1]).joinpath('report.json').read_text())


def main():
    RESULT.mkdir(exist_ok=False);status('bootstrap')
    payload=Path('/content/forge-corpus-source.tar.gz');ROOT.mkdir(exist_ok=False)
    with tarfile.open(payload) as archive:archive.extractall(ROOT,filter='data')
    write(RESULT/'source.json',dict(archive_sha256=sha(payload),manifest=json.loads((ROOT/'payload.json').read_text()),driver_sha256=sha(Path(__file__))))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,compute_cap','--format=csv,noheader,nounits'],text=True).strip()
    fields=[v.strip() for v in gpu.splitlines()[0].split(',')];total_mib=int(fields[1]);compute=fields[2].replace('.','')
    env=os.environ.copy();env['PATH']=str(Path.home()/'.cargo/bin')+':'+env['PATH'];env['CUDA_COMPUTE_CAP']=compute;env['CARGO_BUILD_JOBS']='4'
    if not shutil.which('cargo',path=env['PATH']):
        installer=Path('/content/forge-rustup.sh');urllib.request.urlretrieve('https://sh.rustup.rs',installer)
        run(['sh',installer,'-y','--profile','minimal','--default-toolchain','stable'],env=env)
    write(RESULT/'environment.json',dict(gpu=gpu,cuda_compute_cap=compute,nvcc=subprocess.check_output(['nvcc','--version'],text=True),rust=subprocess.check_output(['rustc','--version'],env=env,text=True).strip(),candle='0.11.0'))
    status('cargo-build-cuda')
    with (RESULT/'build.log').open('w') as log:run(['cargo','build','--locked','--release','-p','fission-forge','--features','cuda'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    forge=str(ROOT/'target/release/forge');config=ROOT/'configs/model/gpu-pilot-5m-8k.json';corpus=ROOT/'data/corpus'
    write(RESULT/'binary.json',dict(sha256=sha(Path(forge)),model=json.loads(config.read_text())))
    def train(name,steps,seed=619,limit=2,resume=None,every=100,timeout=180):
        output=RESULT/name
        cmd=[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',output,'--device','cuda','--steps',str(steps),'--learning-rate','0.001','--seed',str(seed),'--batch-size','1','--attention-chunk','256','--validation-limit',str(limit),'--evaluate-every',str(every),'--checkpoint-every',str(every)]
        if resume:cmd+=['--resume',resume]
        return measure(list(map(str,cmd)),name,env,timeout)
    first=train('preflight-one',1)
    if first['max_training_sequence']!=4878:raise RuntimeError('longest-row preflight identity changed')
    peak=json.loads((RESULT/'preflight-one-resources.json').read_text())['observed_whole_device_peak_mib']
    if peak>total_mib*0.85:raise RuntimeError('preflight uses more than 85% GPU memory; pilot refused')
    resumed=train('preflight-resumed',1,resume=first['final_checkpoint'])
    full=train('preflight-two',2)
    a=Path(resumed['final_checkpoint']);b=Path(full['final_checkpoint'])
    # Safetensors ordering and tensors should be byte-identical on this GPU stack.
    if sha(a/'training.safetensors')!=sha(b/'training.safetensors') or resumed['sampler']!=full['sampler']:raise RuntimeError('GPU resume diverged from uninterrupted updates')
    write(RESULT/'preflight-verification.json',dict(longest_training_sequence=4878,model_parameters=first['parameters'],weights_and_adam_resume_byte_identical=True,sampler_resume_identical=True,observed_gpu_memory_below_85_percent=True))
    # Own corpus only; external Hub samples are never silently mixed into training.
    pilot=train('pilot',480,seed=42,limit=0,every=120,timeout=900)
    write(RESULT/'pilot-summary.json',dict(evidence_scope='bounded_owned_synthetic_GPU_corpus_training_not_functional_accuracy',model_parameters=pilot['parameters'],initial_nll=pilot['initial_validation']['nll'],final_nll=pilot['final_validation']['nll'],train_examples=pilot['train_examples'],validation_examples=pilot['validation_examples'],completed_step=pilot['completed_step'],test_model_examples=pilot['test_model_examples'],final_checkpoint=pilot['final_checkpoint']))
    # Validate GPU inference plumbing on a held-out prompt; raw output is not a score.
    pair=json.loads((corpus/'release/validation.jsonl').read_text().splitlines()[0])
    (RESULT/'generation-raw.json').write_text(json.dumps(pair['input']['raw']));(RESULT/'generation-profile.json').write_text(json.dumps(pair['input']['profile']))
    run([forge,'pcode-compile','--raw',RESULT/'generation-raw.json','--profile',RESULT/'generation-profile.json','--representation','pcode-plus-ssa-v1','--output',RESULT/'generation-input'],env=env,stdout=subprocess.DEVNULL)
    run([forge,'generate','--checkpoint',pilot['final_checkpoint'],'--prompt',RESULT/'generation-input/prompt.txt','--representation','pcode-plus-ssa-v1','--max-new-tokens','128','--output',RESULT/'generation','--device','cuda'],env=env,timeout=120,stdout=subprocess.DEVNULL)


if __name__=='__main__':
    error=None
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        if RESULT.exists():write(RESULT/'failure.json',dict(error=error))
    finally:
        if RESULT.exists():
            status('packaging')
            with tarfile.open(ARCHIVE,'w:gz') as archive:archive.add(RESULT,arcname='results')
        status('done' if error is None else 'failed',state='complete' if error is None else 'failed',error=error,archive_sha256=sha(ARCHIVE) if ARCHIVE.exists() else None)
    if error:raise SystemExit(1)
