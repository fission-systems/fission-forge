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
    selected=[p for p in tracked if p in ['Cargo.toml','Cargo.lock'] or p.startswith('crates/') or p.startswith('configs/model/')]
    # Include explicitly requested new model config before it is committed.
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--gpu',choices=['A100','L4','T4','G4'],default='A100');p.add_argument('--package-only',action='store_true');p.add_argument('--attach',action='store_true',help='Reuse this explicitly named existing session; it will still be stopped when done')
    args=p.parse_args();root=Path.cwd();corpus=args.corpus.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=source_files(root,corpus)
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
        checked(['colab','upload','-s',session,archive,'/content/forge-corpus-source.tar.gz'],timeout=180)
        checked(['colab','upload','-s',session,root/'scripts/colab/corpus_driver.py','/content/forge-corpus-driver.py'],timeout=90)
        warmup=output/'warmup.py';warmup.write_text("print('forge-kernel-ready')\n")
        for attempt in range(4):
            try:
                ready=subprocess.run(['colab','exec','-s',session,'-f',str(warmup)],capture_output=True,text=True,timeout=60)
                if ready.returncode==0:break
            except subprocess.TimeoutExpired:pass
            if attempt==3:raise RuntimeError('kernel did not become ready on the existing session')
            time.sleep(5)
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
