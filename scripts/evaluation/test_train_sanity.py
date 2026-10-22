import json
from pathlib import Path
import tempfile
import unittest
import train_sanity as s

class RawSanityTests(unittest.TestCase):
    def test_byte_identity_and_complete_raw_output_are_required(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);report=dict(checkpoint_sha256='w',prompt_sha256='p',max_new_tokens=128,backend='Cuda { gpu_id: 0 }',tokenizer='byte-v1-260',completion=dict(bytes=[65,66],tokens=[69,70,3]))
            def save(): (p/'report.json').write_text(json.dumps(report))
            save();(p/'completion.bin').write_bytes(b'AB');self.assertEqual(s.raw_completion(p,dict(prompt_sha256='p'),'w',128),b'AB')
            for tokens in [[69,3],[69,3,70],[0,69,70],[69,260]]:
                report['completion']['tokens']=tokens;save()
                with self.assertRaises(ValueError):s.raw_completion(p,dict(prompt_sha256='p'),'w',128)
            report['completion']['tokens']=[69,70,3];save()
            with self.assertRaises(ValueError):s.raw_completion(p,dict(prompt_sha256='changed'),'w',128)
            (p/'completion.bin').write_bytes(b'AB ')
            with self.assertRaises(ValueError):s.raw_completion(p,dict(prompt_sha256='p'),'w',128)
