"""Optional real corpus CLI test for explicit LR changes without checkpoint edits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.getenv('FORGE_CORPUS_FIXTURE'),'set FORGE_CORPUS_FIXTURE for real CPU checkpoint integration')
class LearningRateResumeTests(unittest.TestCase):
    def test_explicit_lr_override_keeps_history_and_rejects_other_changes(self):
        corpus=Path(os.environ['FORGE_CORPUS_FIXTURE']).resolve();forge=Path('target/debug/forge').resolve()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'config.json'
            config.write_text(json.dumps(dict(vocab_size=260,hidden_size=8,heads=1,layers=1,feed_forward_size=16,max_seq_len=8192)))
            def train(name,lr,resume=None,override=False,limit=1):
                cmd=[str(forge),'train-corpus','--config',str(config),'--corpus',str(corpus),'--output',str(root/name),'--steps','1','--learning-rate',str(lr),'--seed','42','--batch-size','1','--attention-chunk','256','--validation-limit',str(limit),'--evaluate-every','1','--checkpoint-every','1']
                if resume:cmd+=['--resume',str(resume)]
                if override:cmd+=['--override-learning-rate']
                return subprocess.run(cmd,capture_output=True,text=True,timeout=120)
            first=train('first',0.001);self.assertEqual(first.returncode,0,first.stderr)
            parent=Path(json.loads(first.stdout)['final_checkpoint']);before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()}
            rejected=train('implicit',0.0001,parent);self.assertNotEqual(rejected.returncode,0);self.assertIn('configuration mismatch',rejected.stderr)
            changed=train('changed',0.0001,parent,True);self.assertEqual(changed.returncode,0,changed.stderr)
            report=json.loads(changed.stdout);self.assertEqual(report['initial_step'],1);self.assertEqual(report['completed_step'],2)
            self.assertEqual(report['learning_rate_transition']['previous_lr'],0.001);self.assertEqual(report['learning_rate_transition']['new_lr'],0.0001)
            checkpoint=Path(report['final_checkpoint']);state=json.loads((checkpoint/'state.json').read_text());self.assertEqual(state['optimizer']['lr'],0.0001)
            self.assertEqual(state['representations'],[json.loads((corpus/'manifest.json').read_text())['representation']])
            self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()},before)
            next_run=train('next',0.0001,checkpoint);self.assertEqual(next_run.returncode,0,next_run.stderr);self.assertEqual(json.loads(next_run.stdout)['completed_step'],3)
            bad=train('settings',0.00001,checkpoint,True,limit=2);self.assertNotEqual(bad.returncode,0);self.assertIn('settings changed',bad.stderr)


if __name__=='__main__':unittest.main()
