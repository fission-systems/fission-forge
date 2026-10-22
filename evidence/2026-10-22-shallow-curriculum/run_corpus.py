"""Package explicit source/corpus files, operate one named Colab GPU, always stop.

Credentials remain on the local CLI. Do not invoke interactive Colab/Drive auth.
The local watchdog survives an assistant tool yield and releases the VM on exit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checked(cmd,timeout=90,**kwargs):return subprocess.run(list(map(str,cmd)),check=True,timeout=timeout,**kwargs)
def source_files(root,corpus):
    tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    selected=[p for p in tracked if p in ['Cargo.toml','Cargo.lock','LICENSE','NOTICE'] or p.startswith('crates/') or p.startswith('configs/model/')]
    selected.append('configs/model/gpu-pilot-5m-8k.json')
    files={name:root/name for name in sorted(set(selected))}
    manifest=json.loads((corpus/'manifest.json').read_text())
    relative=[*manifest['artifacts'],'manifest.json',*[str(p.relative_to(corpus)) for p in (corpus/'release').iterdir() if p.is_file()]]
    for name in relative:
        p=(corpus/name).resolve()
        if not p.is_relative_to(corpus) or not p.is_file():raise ValueError('unsafe corpus artifact path')
        files['data/corpus/'+name]=p
    for name,path in files.items():
        if path.is_symlink() or not path.is_file():raise ValueError('source/payload must be regular files: '+name)
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--gpu',choices=['A100','L4','T4','G4'],default='A100');p.add_argument('--syntax-sanity',action='store_true',help='Run a separately frozen eight-train-example diagnostic before fresh training');p.add_argument('--fresh-epochs',type=int,help='Train a new model on this frozen corpus (1..8 epochs)');p.add_argument('--model-config',type=Path,help='Explicit model configuration for a fresh corpus job');p.add_argument('--tokenizer',type=Path,help='Train-only learned BPE specification for a fresh model');p.add_argument('--fresh-learning-rate',type=float,default=0.0003);p.add_argument('--generation-checkpoint',type=Path,help='Generate every validation target with frozen verified weights; no training');p.add_argument('--diagnostic-checkpoint',type=Path,help='Evaluate a frozen checkpoint with swapped validation inputs; no updates');p.add_argument('--resume',type=Path,help='Continue a verified schema-4 local corpus checkpoint');p.add_argument('--additional-steps',type=int,default=1920);p.add_argument('--resume-learning-rate',type=float,help='Explicitly fork the resume LR while preserving Adam history');p.add_argument('--package-only',action='store_true');p.add_argument('--attach',action='store_true',help='Reuse this explicitly named existing session; it will still be stopped when done')
    args=p.parse_args();root=Path.cwd();corpus=args.corpus.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if sum([args.fresh_epochs is not None,args.resume is not None,args.diagnostic_checkpoint is not None,args.generation_checkpoint is not None])>1:raise ValueError('fresh, resume, and diagnostic modes are mutually exclusive')
    if args.fresh_epochs is not None and args.resume is not None:raise ValueError('fresh training cannot resume an old corpus checkpoint')
    if args.syntax_sanity and args.fresh_epochs is None:raise ValueError('syntax sanity requires a fresh shallow corpus job')
    if args.model_config and args.fresh_epochs is None:raise ValueError('model config override requires a fresh job')
    if args.tokenizer and args.fresh_epochs is None:raise ValueError('BPE requires a fresh training job')
    if args.resume_learning_rate is not None and args.resume is None:raise ValueError('resume LR requires --resume')
    files=source_files(root,corpus)
    if args.diagnostic_checkpoint or args.generation_checkpoint:
        resume=(args.diagnostic_checkpoint or args.generation_checkpoint).resolve();state=json.loads((resume/'state.json').read_text())
        if state.get('schema_version') not in (4,5) or sha(resume/'training.safetensors')!=state.get('checkpoint_sha256'):raise ValueError('diagnostic checkpoint integrity mismatch')
        if state['corpus']['corpus_lock_sha256']!=sha(corpus/'release/lock.json'):raise ValueError('diagnostic corpus lock mismatch')
        for name in ['state.json','training.safetensors']:files['resume/'+name]=resume/name
        model=output/'model.json';model.write_text(json.dumps(state['model'],indent=2)+'\n');files['model.json']=model
        job=output/'job.json';job.write_text(json.dumps(dict(mode='generate' if args.generation_checkpoint else 'diagnostic',representation=state['representations'][0],generation_max_new_tokens=512,corpus_lock_sha256=state['corpus']['corpus_lock_sha256'],baseline_step=state['step'],baseline_checkpoint_sha256=state['checkpoint_sha256'],checkpoint_state_sha256=sha(resume/'state.json'),optimizer_updates=0,expected_validation_examples=130,control_rotation=17),indent=2)+'\n');files['job.json']=job
        if args.generation_checkpoint:files['fresh_job.py']=root/'scripts/colab/fresh_job.py'
    if args.resume:
        if not 1<=args.additional_steps<=1920:raise ValueError('continuation requires 1..1920 additional updates per bounded job')
        resume=args.resume.resolve();state=json.loads((resume/'state.json').read_text())
        if state.get('schema_version')!=4 or sha(resume/'training.safetensors')!=state.get('checkpoint_sha256'):raise ValueError('resume checkpoint integrity mismatch')
        if state['corpus']['corpus_lock_sha256']!=sha(corpus/'release/lock.json'):raise ValueError('resume corpus lock mismatch')
        if args.resume_learning_rate is not None and not 0<args.resume_learning_rate<=0.001:raise ValueError('resume LR must be >0 and <=0.001')
        for name in ['state.json','training.safetensors']:files['resume/'+name]=resume/name
        job=output/'job.json';job.write_text(json.dumps(dict(mode='continue',additional_steps=args.additional_steps,baseline_step=state['step'],baseline_checkpoint_sha256=state['checkpoint_sha256'],generation_max_new_tokens=256,functional_targets='all_validation',resume_learning_rate=args.resume_learning_rate),indent=2)+'\n');files['job.json']=job
    if args.fresh_epochs is not None:
        if not 1<=args.fresh_epochs<=8 or not 0<args.fresh_learning_rate<=0.001:raise ValueError('fresh epochs/LR out of bounded range')
        train_count=len((corpus/'release/train.jsonl').read_text().splitlines())
        job=output/'job.json';job.write_text(json.dumps(dict(mode='fresh',representation=json.loads((corpus/'manifest.json').read_text()).get('representation','pcode-plus-ssa-v1'),epochs=args.fresh_epochs,steps=train_count*args.fresh_epochs,train_examples=train_count,learning_rate=args.fresh_learning_rate,evaluate_every=256,generation_max_new_tokens=512,selection='lowest_full_validation_NLL_before_functional_scoring',corpus_lock_sha256=sha(corpus/'release/lock.json')),indent=2)+'\n');files['job.json']=job
        files['fresh_job.py']=root/'scripts/colab/fresh_job.py'
        if args.syntax_sanity:
            import syntax_sanity
            job_data=json.loads(job.read_text());job_data['syntax_sanity']=syntax_sanity.prepare(corpus);job.write_text(json.dumps(job_data,indent=2)+'\n');files['syntax_sanity.py']=root/'scripts/colab/syntax_sanity.py'
        if args.tokenizer:
            tokenizer=args.tokenizer.resolve();spec=json.loads(tokenizer.read_text())
            if spec['train_split_sha256']!=sha(corpus/'release/train.jsonl'):raise ValueError('BPE train split changed')
            job_data=json.loads(job.read_text());job_data.update(tokenizer='lexical-byte-bpe-v1',tokenizer_file_sha256=sha(tokenizer),tokenizer_sha256=hashlib.sha256(json.dumps(spec,separators=(',',':')).encode()).hexdigest(),max_output_bytes=512)
            job.write_text(json.dumps(job_data,indent=2)+'\n');files['tokenizer.json']=tokenizer
            config=json.loads((args.model_config or root/'configs/model/gpu-pilot-5m-8k.json').read_text());config['vocab_size']=260+len(spec['merges'])
            model=output/'model.json';model.write_text(json.dumps(config,indent=2)+'\n');files['model.json']=model

    if args.model_config and not args.tokenizer:files['model.json']=args.model_config.resolve()
    if args.fresh_epochs is not None and 'model.json' in files:
        job_data=json.loads(job.read_text());job_data['model_sha256']=sha(files['model.json']);job.write_text(json.dumps(job_data,indent=2)+'\n')
    metadata=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),files={name:sha(path) for name,path in files.items()},corpus_lock_sha256=sha(corpus/'release/lock.json'),credentials_uploaded=False)
    metadata_path=output/'payload.json';metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    archive=output/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in files.items():tar.add(path,arcname=name,recursive=False)
        tar.add(metadata_path,arcname='payload.json')
    if args.package_only:print(archive);return
    session=args.session;started=time.monotonic();allocated=False
    try:
        if args.attach:
            checked(['colab','status','-s',session]);allocated=True
        else:
            checked(['colab','new','-s',session,'--gpu',args.gpu],timeout=180);allocated=True
        pieces=[]
        if archive.stat().st_size>8*1024*1024:
            with archive.open('rb') as incoming:
                while data:=incoming.read(4*1024*1024):
                    part=output/('source.part-%04d'%len(pieces));part.write_bytes(data);pieces.append(part)
            for part in pieces:checked(['colab','upload','-s',session,part,'/content/'+part.name],timeout=90)
        else:checked(['colab','upload','-s',session,archive,'/content/forge-corpus-source.tar.gz'],timeout=180)
        checked(['colab','upload','-s',session,root/'scripts/colab/corpus_driver.py','/content/forge-corpus-driver.py'],timeout=90)
        warmup=output/'warmup.py';warmup.write_text("print('forge-kernel-ready')\n")
        for attempt in range(4):
            try:
                ready=subprocess.run(['colab','exec','-s',session,'-f',str(warmup)],capture_output=True,text=True,timeout=60)
                if ready.returncode==0:break
            except subprocess.TimeoutExpired:pass
            if attempt==3:raise RuntimeError('kernel did not become ready on the existing session')
            time.sleep(5)
        if pieces:
            assembly=output/'assemble.py';assembly.write_text("import hashlib,pathlib\nparts="+repr([p.name for p in pieces])+"\noutput=pathlib.Path('/content/forge-corpus-source.tar.gz')\nwith output.open('wb') as out:\n    for name in parts:out.write(pathlib.Path('/content',name).read_bytes())\nassert hashlib.sha256(output.read_bytes()).hexdigest()=="+repr(sha(archive))+"\nprint('archive assembled and SHA verified')\n")
            checked(['colab','exec','-s',session,'-f',assembly],timeout=90)
        launch=output/'launch.py';launch.write_text("""import subprocess,sys,pathlib,json,os
pidfile=pathlib.Path('/content/forge-corpus-pid.json')
statusfile=pathlib.Path('/content/forge-corpus-status.json')
active=[]
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid']
    try:os.kill(pid,0);active=[pid]
    except ProcessLookupError:pass
if not active:
    found=subprocess.run(['pgrep','-f','[f]orge-corpus-driver.py'],capture_output=True,text=True)
    active=[int(p) for p in found.stdout.split()]
if active:
    print({'already_running':active})
elif statusfile.exists() and json.loads(statusfile.read_text()).get('state') in ('complete','failed'):
    print({'already_terminal':True})
else:
    log=open('/content/forge-corpus-driver.log','w')
    p=subprocess.Popen([sys.executable,'/content/forge-corpus-driver.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(json.dumps({'pid':p.pid}))
    print({'pid':p.pid})
""")
        for attempt in range(4):
            try:
                launched=subprocess.run(['colab','exec','-s',session,'-f',str(launch)],capture_output=True,text=True,timeout=60)
                if launched.returncode==0:print(launched.stdout,flush=True);break
            except subprocess.TimeoutExpired:pass
            if attempt==3:raise RuntimeError('could not confirm idempotent driver launch on existing session')
            time.sleep(5)
        last=None;failures=0
        poll=output/'poll.py';poll.write_text("""import pathlib,json,os
p=pathlib.Path('/content/forge-corpus-status.json')
r=json.loads(p.read_text()) if p.exists() else {'state':'starting'}
pidfile=pathlib.Path('/content/forge-corpus-pid.json')
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid']
    try:os.kill(pid,0);r['driver_alive']=True
    except ProcessLookupError:r['driver_alive']=False
log=pathlib.Path('/content/forge-corpus-results/build.log')
if log.exists():r['build_tail']=log.read_text()[-1200:].splitlines()[-3:]
print(json.dumps(r))
""")
        while time.monotonic()-started<1800:
            try:
                result=subprocess.run(['colab','exec','-s',session,'-f',str(poll)],capture_output=True,text=True,timeout=50)
            except subprocess.TimeoutExpired:
                print('Status observation timed out; retrying the same session',flush=True)
                continue
            (output/'last-poll.txt').write_text(result.stdout+result.stderr)
            try:
                text=result.stdout;status=json.loads(text[text.index('{'):text.rindex('}')+1]);failures=0
            except (ValueError,KeyError):
                failures+=1
                if failures==4:print('Status unavailable repeatedly; retaining session observation until budget expires',flush=True)
                time.sleep(5);continue
            if status.get('phase')!=last:print(json.dumps(status),flush=True);last=status.get('phase')
            if status['state'] in ('complete','failed'):
                checked(['colab','download','-s',session,'/content/forge-corpus-results.tar.gz',output/'results.tar.gz'],timeout=240)
                checked(['colab','download','-s',session,'/content/forge-corpus-status.json',output/'status.json'])
                terminal=json.loads((output/'status.json').read_text())
                if sha(output/'results.tar.gz')!=terminal.get('archive_sha256'):raise ValueError('downloaded result archive digest mismatch')
                with tarfile.open(output/'results.tar.gz') as tar:tar.extractall(output,filter='data')
                if status['state']!='complete':raise RuntimeError(status.get('error','remote corpus pilot failed'))
                return
            time.sleep(5)
        raise TimeoutError('GPU session exceeded the 30-minute allocation budget')
    finally:
        if allocated:
            try:
                for remote,name in [('/content/forge-corpus-driver.log','driver.log'),('/content/forge-corpus-status.json','final-status.json')]:
                    try:
                        subprocess.run(['colab','download','-s',session,remote,str(output/name)],timeout=60,capture_output=True)
                    except subprocess.TimeoutExpired:
                        print('Diagnostic download timed out: '+name,flush=True)
            finally:
                checked(['colab','stop','-s',session],timeout=90)



if __name__=='__main__':main()
