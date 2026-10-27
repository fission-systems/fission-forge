"""Dataset boundaries and metadata handling; no network or external code execution."""
import io
import sys
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zstandard
import hf_inventory
import exebench_probe


class IntakeTests(unittest.TestCase):
    def test_reserved_evaluation_never_reads_viewer_or_examples(self):
        calls=[]
        def fetch(url,json_body=True):
            calls.append(url)
            return {'sha':'a'*40,'siblings':[]} if json_body else '# dataset card'
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(hf_inventory,'fetch',side_effect=fetch):
            r=hf_inventory.inspect(dict(repo='owner/eval',revision='a'*40,policy='evaluation_reserved'),Path(folder))
        self.assertFalse(r['content_sampled']);self.assertFalse(r['training_approved'])
        self.assertFalse(any('datasets-server' in u for u in calls))
    def test_archive_probe_keeps_license_and_ref_unknown_without_admission(self):
        rows=[dict(text=dict(path='o/p/f.c',ref='master',func_def='double f(double a){return a;}',fname='f',real_io_pairs=[{}]),meta={}),dict(text=dict(path='o/q/g.c',ref='b'*40,func_def='unsigned f(unsigned a){return a;}',fname='g'),meta={'license':'MIT'})]
        data=zstandard.ZstdCompressor().compress(''.join(json.dumps(r)+'\n' for r in rows).encode())
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.tar.gz'
            with tarfile.open(p,'w:gz') as tar:
                member=tarfile.TarInfo('../../not-extracted.jsonl.zst');member.size=len(data);tar.addfile(member,io.BytesIO(data))
            sample=exebench_probe.inspect(p,2)
            self.assertFalse((Path(folder).parent/'not-extracted.jsonl.zst').exists())
        self.assertIsNone(sample[0]['source_license_metadata']);self.assertFalse(sample[0]['upstream_ref_is_commit']);self.assertTrue(sample[0]['contains_floating_types'])
        self.assertEqual(sample[1]['source_license_metadata'],'MIT');self.assertTrue(sample[1]['upstream_ref_is_commit'])
        self.assertTrue(all(not r['training_approved'] for r in sample))
    def test_archive_digest_mismatch_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'bad.tar.gz';archive.write_bytes(b'incorrect archive')
            output=Path(folder)/'probe'
            with mock.patch.object(sys,'argv',['probe','--archive',str(archive),'--output',str(output),'--expected-sha256','0'*64]),self.assertRaisesRegex(ValueError,'digest differs'):
                exebench_probe.main()
            self.assertFalse(output.exists())
    def test_candidate_registry_reserves_both_external_evaluations(self):
        cfg=json.loads(Path('configs/data/hf-candidates-v1.json').read_text());sources={s['repo']:s for s in cfg['sources']}
        for repo in ['noelo-lab/decbench-dataset','LLM4Binary/decompile-eval','sjkim1127/decbench-cpp-raw-corpus','sjkim1127/decbench-cpp-qualification','sjkim1127/decbench-multilang-corpus','sjkim1127/decbench-cpp-dataset-unofficial']:
            self.assertEqual(sources[repo]['policy'],'evaluation_reserved')
        self.assertFalse(cfg['training_admission']['approved'])


if __name__=='__main__':unittest.main()
