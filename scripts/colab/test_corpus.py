"""Exercise local lease control without allocating a real VM or spending units."""
import json
import hashlib
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
    elif remote.endswith('.json'):destination.write_text(json.dumps({'state':'complete','phase':'done','archive_sha256':'0'*64 if os.environ.get('FORGE_BAD_DIGEST') else hashlib.sha256((root/'fake.tar.gz').read_bytes()).hexdigest()}))
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
    def test_resume_packages_only_verified_checkpoint_and_records_plan(self):
        resume=self.root/'resume';resume.mkdir();weights=b'owned test checkpoint'
        (resume/'training.safetensors').write_bytes(weights)
        state=dict(schema_version=4,step=480,checkpoint_sha256=hashlib.sha256(weights).hexdigest(),corpus=dict(corpus_lock_sha256=hashlib.sha256(b'{}').hexdigest()))
        (resume/'state.json').write_text(json.dumps(state))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--resume',str(resume),'--package-only']):run_corpus.main()
        with tarfile.open(self.root/'job/source.tar.gz') as tar:
            plan=json.load(tar.extractfile('job.json'))
            self.assertEqual(plan['baseline_step'],480);self.assertEqual(plan['additional_steps'],1920)
            self.assertEqual(tar.extractfile('resume/training.safetensors').read(),weights)
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_diagnostic_packages_exact_model_and_unchanged_checkpoint_without_training(self):
        resume=self.root/'resume';resume.mkdir();weights=b'frozen diagnostic weights'
        (resume/'training.safetensors').write_bytes(weights)
        state=dict(schema_version=5,step=768,representations=['pcode-plus-ssa-v1'],model=dict(vocab_size=698,rotary_positions=True),checkpoint_sha256=hashlib.sha256(weights).hexdigest(),corpus=dict(corpus_lock_sha256=hashlib.sha256(b'{}').hexdigest()))
        (resume/'state.json').write_text(json.dumps(state))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--diagnostic-checkpoint',str(resume),'--package-only']):run_corpus.main()
        with tarfile.open(self.root/'job/source.tar.gz') as tar:
            plan=json.load(tar.extractfile('job.json'))
            self.assertEqual(plan['mode'],'diagnostic');self.assertEqual(plan['optimizer_updates'],0)
            self.assertEqual(json.load(tar.extractfile('model.json')),state['model'])
            self.assertEqual(tar.extractfile('resume/training.safetensors').read(),weights)
            self.assertNotIn('fresh_job.py',tar.getnames())
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_generation_only_packages_selected_checkpoint_without_training(self):
        resume=self.root/'resume';resume.mkdir();weights=b'frozen generation weights'
        (resume/'training.safetensors').write_bytes(weights)
        helper=self.root/'scripts/colab/fresh_job.py';helper.parent.mkdir(parents=True);helper.write_text('# generation helper fixture\n')
        state=dict(schema_version=5,step=768,representations=['pcode-plus-ssa-return-u32-v1'],model=dict(vocab_size=500),checkpoint_sha256=hashlib.sha256(weights).hexdigest(),corpus=dict(corpus_lock_sha256=hashlib.sha256(b'{}').hexdigest()))
        (resume/'state.json').write_text(json.dumps(state))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--generation-checkpoint',str(resume),'--package-only']):run_corpus.main()
        with tarfile.open(self.root/'job/source.tar.gz') as tar:
            plan=json.load(tar.extractfile('job.json'))
            self.assertEqual(plan['mode'],'generate');self.assertEqual(plan['optimizer_updates'],0)
            self.assertEqual(plan['representation'],state['representations'][0]);self.assertEqual(plan['generation_max_new_tokens'],512)
            self.assertIn('fresh_job.py',tar.getnames());self.assertNotIn('steps',plan)
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_corrupt_diagnostic_refused_before_runtime_allocation(self):
        resume=self.root/'resume';resume.mkdir();(resume/'training.safetensors').write_bytes(b'changed')
        (resume/'state.json').write_text(json.dumps(dict(schema_version=5,checkpoint_sha256='0'*64)))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--diagnostic-checkpoint',str(resume)]),self.assertRaisesRegex(ValueError,'integrity mismatch'):run_corpus.main()
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_corrupt_resume_refused_before_runtime_allocation(self):
        resume=self.root/'resume';resume.mkdir();(resume/'training.safetensors').write_bytes(b'changed')
        (resume/'state.json').write_text(json.dumps(dict(schema_version=4,checkpoint_sha256='0'*64)))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--resume',str(resume)]),self.assertRaisesRegex(ValueError,'integrity mismatch'):run_corpus.main()
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_bad_result_archive_digest_still_releases_runtime(self):
        with self.assertRaisesRegex(ValueError,'digest mismatch'):self.invoke({'FORGE_BAD_DIGEST':'1'})
        self.assertEqual(sum(c[0]=='stop' for c in self.calls()),1)
        self.assertFalse((self.root/'job/results').exists())
    def test_large_archive_upload_is_chunked_and_assembled_before_launch(self):
        large=self.root/'crates/large.bin';large.parent.mkdir();large.write_bytes(os.urandom(9*1024*1024))
        subprocess.run(['git','add',str(large)],check=True)
        self.invoke();calls=self.calls()
        uploads=[c for c in calls if c[0]=='upload' and '/source.part-' in c[-1]]
        self.assertGreater(len(uploads),1)
        self.assertTrue(all(Path(c[-2]).stat().st_size<=4*1024*1024 for c in uploads))
        archive=self.root/'job/source.tar.gz'
        self.assertEqual(b''.join(Path(c[-2]).read_bytes() for c in uploads),archive.read_bytes())
        assemble=next(i for i,c in enumerate(calls) if c[0]=='exec' and c[-1].endswith('assemble.py'))
        launch=next(i for i,c in enumerate(calls) if c[0]=='exec' and c[-1].endswith('launch.py'))
        self.assertLess(assemble,launch)
        self.assertIn(hashlib.sha256(archive.read_bytes()).hexdigest(),(self.root/'job/assemble.py').read_text())
    def test_fresh_plan_binds_release_count_without_old_checkpoint(self):
        (self.corpus/'release/train.jsonl').write_text('{}\n{}\n{}\n')
        helper=self.root/'scripts/colab/fresh_job.py';helper.parent.mkdir(parents=True);helper.write_text('# fixture helper\n')
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--fresh-epochs','6','--package-only']):run_corpus.main()
        with tarfile.open(self.root/'job/source.tar.gz') as tar:
            plan=json.load(tar.extractfile('job.json'));self.assertEqual(plan['steps'],18);self.assertEqual(plan['mode'],'fresh');self.assertEqual(plan['generation_max_new_tokens'],512)
            self.assertIn('fresh_job.py',tar.getnames());self.assertNotIn('resume/state.json',tar.getnames())
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_bpe_package_binds_train_split_and_adapts_model_vocabulary(self):
        (self.corpus/'release/train.jsonl').write_text('{}\n')
        helper=self.root/'scripts/colab/fresh_job.py';helper.parent.mkdir(parents=True);helper.write_text('# fixture\n')
        spec=dict(algorithm='lexical-byte-bpe-v1',train_split_sha256=run_corpus.sha(self.corpus/'release/train.jsonl'),source_weight=8,merges=[[4,5]])
        tokenizer=self.root/'tokenizer.json';tokenizer.write_text(json.dumps(spec))
        config=self.root/'rope.json';config.write_text(json.dumps(dict(vocab_size=260,rotary_positions=True)))
        with mock.patch.object(sys,'argv',['run_corpus.py','--corpus',str(self.corpus),'--output',str(self.root/'job'),'--session','unit-gpu','--fresh-epochs','8','--tokenizer',str(tokenizer),'--model-config',str(config),'--package-only']):run_corpus.main()
        with tarfile.open(self.root/'job/source.tar.gz') as tar:
            plan=json.load(tar.extractfile('job.json'));model=json.load(tar.extractfile('model.json'))
            self.assertEqual(plan['max_output_bytes'],512);self.assertEqual(plan['tokenizer_file_sha256'],run_corpus.sha(tokenizer))
            self.assertEqual(model['vocab_size'],261);self.assertEqual(plan['steps'],8)
            self.assertTrue(model['rotary_positions']);self.assertEqual(plan['model_sha256'],hashlib.sha256(tar.extractfile('model.json').read()).hexdigest())
        self.assertFalse((self.root/'calls.jsonl').exists())
    def test_launch_failure_still_releases_runtime(self):
        with self.assertRaisesRegex(RuntimeError,'confirm idempotent'):self.invoke({'FORGE_LAUNCH_FAIL':'1'})
        self.assertEqual(sum(c[0]=='stop' for c in self.calls()),1)


if __name__=='__main__':unittest.main()
