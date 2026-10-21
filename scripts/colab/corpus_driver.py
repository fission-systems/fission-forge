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
import sys
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
    if (ROOT/'model.json').exists():config=ROOT/'model.json'
    write(RESULT/'binary.json',dict(sha256=sha(Path(forge)),model=json.loads(config.read_text())))
    def train(name,steps,seed=619,limit=2,resume=None,every=100,timeout=180,learning_rate=0.001,override_learning_rate=False):
        output=RESULT/name
        cmd=[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',output,'--device','cuda','--steps',str(steps),'--learning-rate',str(learning_rate),'--seed',str(seed),'--batch-size','1','--attention-chunk','256','--validation-limit',str(limit),'--evaluate-every',str(every),'--checkpoint-every',str(every)]
        if (ROOT/'tokenizer.json').exists():cmd+=['--tokenizer',ROOT/'tokenizer.json']
        if resume:cmd+=['--resume',resume]
        if override_learning_rate:cmd+=['--override-learning-rate']
        return measure(list(map(str,cmd)),name,env,timeout)
    jobpath=ROOT/'job.json'
    if jobpath.exists() and json.loads(jobpath.read_text())['mode']=='fresh':
        sys.path.insert(0,str(ROOT));import fresh_job
        fresh_job.run(json.loads(jobpath.read_text()),ROOT,RESULT,forge,config,corpus,env,train,write,sha,status,START,total_mib)
    elif jobpath.exists():
        job=json.loads(jobpath.read_text());resume=ROOT/'resume'
        if sha(resume/'training.safetensors')!=job['baseline_checkpoint_sha256']:raise RuntimeError('resume payload changed')
        write(RESULT/'continuation-plan.json',job)
        pairs=[json.loads(line) for line in (corpus/'release/validation.jsonl').read_text().splitlines()]
        prompts=RESULT/'validation-prompts';prompts.mkdir()
        for pair in pairs:
            folder=prompts/pair['id'];folder.mkdir()
            write(folder/'raw.json',pair['input']['raw']);write(folder/'profile.json',pair['input']['profile'])
            run([forge,'pcode-compile','--raw',folder/'raw.json','--profile',folder/'profile.json','--representation','pcode-plus-ssa-v1','--output',folder/'input'],env=env,stdout=subprocess.DEVNULL)
        def generate_all(name,checkpoint):
            status(name);output=RESULT/name;output.mkdir();records=[]
            for i,pair in enumerate(pairs):
                if time.monotonic()-START>1500:raise TimeoutError('global generation budget exceeded')
                folder=output/pair['id'];prompt=prompts/pair['id']/'input/prompt.txt'
                command=[forge,'generate','--checkpoint',checkpoint,'--prompt',prompt,'--representation','pcode-plus-ssa-v1','--max-new-tokens',str(job['generation_max_new_tokens']),'--output',folder,'--device','cuda']
                try:
                    generated=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True,timeout=30)
                    record=dict(id=pair['id'],exit_code=generated.returncode,timed_out=False,diagnostic=generated.stderr[-2000:],prompt_sha256=sha(prompt),checkpoint_sha256=sha(checkpoint/'training.safetensors'))
                except subprocess.TimeoutExpired:
                    record=dict(id=pair['id'],exit_code=None,timed_out=True,prompt_sha256=sha(prompt),checkpoint_sha256=sha(checkpoint/'training.safetensors'))
                records.append(record)
                if (i+1)%20==0:status(name,completed=i+1,total=len(pairs))
            write(output/'index.json',dict(records=records,expected_targets=len(pairs),max_new_tokens=job['generation_max_new_tokens'],device='cuda',test_model_examples=0))
        generate_all('baseline-generations',resume)
        pilot=train('continuation',job['additional_steps'],seed=42,limit=0,every=120,resume=resume,timeout=900,learning_rate=job.get('resume_learning_rate') or 0.001,override_learning_rate=job.get('resume_learning_rate') is not None)
        if pilot['initial_step']!=job['baseline_step'] or pilot['completed_step']!=job['baseline_step']+job['additional_steps']:raise RuntimeError('continuation step mismatch')
        generate_all('continued-generations',Path(pilot['final_checkpoint']))
        write(RESULT/'continuation-summary.json',dict(initial_step=pilot['initial_step'],completed_step=pilot['completed_step'],initial_nll=pilot['initial_validation']['nll'],final_nll=pilot['final_validation']['nll'],best_validation=pilot['best_validation'],external_training_rows=0,test_model_examples=0))
    else:
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
            keep_weights=set()
            report=RESULT/'fresh/report.json' if (RESULT/'fresh/report.json').exists() else RESULT/'continuation/report.json'
            if report.exists():
                info=json.loads(report.read_text());keep_weights.add(Path(info['final_checkpoint']).name)
                if info.get('best_validation'):keep_weights.add('step-%08d'%info['best_validation']['step'])
            def select(item):
                if keep_weights and item.name.endswith('/training.safetensors') and Path(item.name).parent.name not in keep_weights:return None
                return item
            with tarfile.open(ARCHIVE,'w:gz') as archive:archive.add(RESULT,arcname='results',filter=select)
        status('done' if error is None else 'failed',state='complete' if error is None else 'failed',error=error,archive_sha256=sha(ARCHIVE) if ARCHIVE.exists() else None)
    if error:raise SystemExit(1)
