"""Integrity and denominator gates for the Colab functional comparison."""
import json
from pathlib import Path
import tempfile
import unittest
import score_gpu


class GpuScoreTests(unittest.TestCase):
    def test_missing_duplicate_or_changed_prompt_cannot_shrink_denominator(self):
        targets=[dict(id='a',prompt_sha256='p'),dict(id='b',prompt_sha256='q')]
        index=dict(expected_targets=2,device='cuda',max_new_tokens=256,test_model_examples=0,records=[dict(id='a',prompt_sha256='p',checkpoint_sha256='w'),dict(id='b',prompt_sha256='q',checkpoint_sha256='w')])
        self.assertEqual(len(score_gpu.validate_index(index,targets,'w')),2)
        for change in ['missing','duplicate','prompt','checkpoint']:
            altered=json.loads(json.dumps(index))
            if change=='missing':altered['records'].pop()
            elif change=='duplicate':altered['records'][1]['id']='a'
            elif change=='prompt':altered['records'][1]['prompt_sha256']='changed'
            else:altered['records'][1]['checkpoint_sha256']='different'
            with self.assertRaises(ValueError):score_gpu.validate_index(altered,targets,'w')
    def test_bpe_raw_bytes_are_reconstructed_from_bound_checkpoint_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);spec=dict(algorithm='lexical-byte-bpe-v1',train_split_sha256='a'*64,source_weight=8,merges=[[69,70]])
            digest=score_gpu.f.sha(json.dumps(spec,separators=(',',':')).encode())
            report=dict(prompt_sha256='p',checkpoint_sha256='w',max_new_tokens=512,max_output_bytes=512,backend='Cuda { gpu_id: 0 }',decoding='greedy_eos_and_bpe_tokens_kv_cache_byte_budget',tokenizer='lexical-byte-bpe-v1',tokenizer_sha256=digest,completion=dict(tokens=[260,3],bytes=[65,66]))
            (folder/'report.json').write_text(json.dumps(report));(folder/'completion.bin').write_bytes(b'AB')
            self.assertEqual(score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w',512,spec),b'AB')
            report['completion']['tokens']=[69,3];(folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'token/byte mismatch'):score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w',512,spec)
    def test_raw_completion_keeps_bytes_and_rejects_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=b'not C\xff'
            report=dict(prompt_sha256='p',checkpoint_sha256='w',max_new_tokens=256,backend='Cuda { gpu_id: 0 }',decoding='greedy_eos_and_bytes_only_kv_cache',completion=dict(bytes=list(raw)))
            (folder/'report.json').write_text(json.dumps(report));(folder/'completion.bin').write_bytes(raw)
            self.assertEqual(score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w'),raw)
            report['max_new_tokens']=512;(folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'identity changed'):score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w')
            self.assertEqual(score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w',512),raw)
            report['max_new_tokens']=256;(folder/'report.json').write_text(json.dumps(report))
            (folder/'completion.bin').write_bytes(b'unsigned f(unsigned a,unsigned b){return a;}')
            with self.assertRaisesRegex(ValueError,'raw completion bytes'):score_gpu.raw_completion(folder,dict(prompt_sha256='p'),'w')


if __name__=='__main__':unittest.main()
