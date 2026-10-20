"""Exercise local lease control without allocating a real VM or spending units."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import run_corpus


FAKE='''#!/usr/bin/env python3
import hashlib,json,os,pathlib,shutil,sys
args=sys.argv[1:];root=pathlib.Path(os.environ['FORGE_COLAB_TEST'])
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')
command=args[0]
if command=='exec':
    path=pathlib.Path(args[args.index('-f')+1])
    if path.name=='poll.py':
        countfile=root/'poll-count';count=int(countfile.read_text())+1 if countfile.exists() else 1;countfile.write_text(str(count))
        if count<=int(os.environ.get('FORGE_POLL_FAILURES','0')):print('Connection lost',file=sys.stderr);raise SystemExit(1)
        print(json.dumps({'state':'complete','phase':'done'}))
    elif path.name=='launch.py':
        if os.environ.get('FORGE_LAUNCH_FAIL'):raise SystemExit(1)
        print(json.dumps({'pid':1234}))
    else:print('ready')
elif command=='download':
    remote,destination=args[-2:];destination=pathlib.Path(destination)
    if remote.endswith('.tar.gz'):shutil.copyfile(root/'fake.tar.gz',destination)
    elif remote.endswith('.json'):destination.write_text(json.dumps({'state':'complete','phase':'done','archive_sha256':hashlib.sha256((root/'fake.tar.gz').read_bytes()).hexdigest()}))
    else:destination.write_text('fake log')
else:print('ok')
'''


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.previous=Path.cwd();os.chdir(self.root)
        subprocess.run(['git','init','-q'],check=True)
        for name,data in [('Cargo.toml','[workspace]\n'),('Cargo.lock',''),('configs/model/gpu-pilot-5m-8k.json','{}')]:
            p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
        subprocess.run(['git','add','.'],check=True)
        subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
        corpus=self.root/'corpus';(corpus/'release').mkdir(parents=True)
        (corpus/'manifest.json').write_text(json.dumps({'artifacts':{}}));(corpus/'release/lock.json').write_text('{}')
        self.corpus=corpus
        fakebin=self.root/'bin';fakebin.mkdir();exe=fakebin/'colab';exe.write_text(FAKE);exe.chmod(0o755)
        payload=self.root/'payload/results';payload.mkdir(parents=True);(payload/'proof.txt').write_text('recovered')
        with tarfile.open(self.root/'fake.tar.gz','w:gz') as tar:tar.add(payload,arcname='results')
        self.env={'PATH':str(fakebin)+':'+os.environ['PATH'],'FORGE_COLAB_TEST':str(self.root)}
    def tearDown(self):os.chdir(self.previous);self.temp.cleanup()
    def invoke(self,extra=None):
        with mock.patch.dict(os.environ,self.env| (extra or {})),mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu']),mock.patch.object(run_corpus.time,'sleep',return_value=None):
            run_corpus.main()
    def calls(self):return [json.loads(line) for line in (self.root/'calls.jsonl').read_text().splitlines()]
    def test_success_downloads_results_and_releases_exactly_once(self):
        self.invoke();calls=self.calls()
        self.assertEqual(sum(c[0]=='new' for c in calls),1);self.assertEqual(sum(c[0]=='stop' for c in calls),1)
        self.assertEqual((self.root/'job/results/proof.txt').read_text(),'recovered')
        self.assertFalse(json.loads((self.root/'job/payload.json').read_text())['credentials_uploaded'])
    def test_transient_observation_failures_reuse_same_session_without_relaunch(self):
        self.invoke({'FORGE_POLL_FAILURES':'4'});calls=self.calls()
        self.assertEqual(sum(c[0]=='new' for c in calls),1)
        self.assertEqual(sum(c[0]=='exec' and c[-1].endswith('launch.py') for c in calls),1)
        self.assertEqual(sum(c[0]=='stop' for c in calls),1)
        self.assertEqual((self.root/'poll-count').read_text(),'5')
    def test_launch_observation_timeout_retries_same_session(self):
        original=run_corpus.subprocess.run;seen=[]
        def run(cmd,*args,**kwargs):
            if cmd[0]=='colab' and cmd[1]=='exec' and str(cmd[-1]).endswith('launch.py'):
                seen.append(cmd)
                if len(seen)==1:raise subprocess.TimeoutExpired(cmd,60)
            return original(cmd,*args,**kwargs)
        with mock.patch.object(run_corpus.subprocess,'run',side_effect=run):self.invoke()
        self.assertEqual(len(seen),2)
        self.assertEqual(sum(c[0]=='new' for c in self.calls()),1)
        self.assertEqual(sum(c[0]=='stop' for c in self.calls()),1)
    def test_launch_failure_still_releases_runtime(self):
        with self.assertRaisesRegex(RuntimeError,'confirm idempotent'):self.invoke({'FORGE_LAUNCH_FAIL':'1'})
        self.assertEqual(sum(c[0]=='stop' for c in self.calls()),1)


if __name__=='__main__':unittest.main()
