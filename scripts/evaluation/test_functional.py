"""Real Clang + capability-free Wasmtime contract regressions, not model scores."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
import functional


@unittest.skipUnless(importlib.util.find_spec('wasmtime'), 'install requirements-functional.txt')
class FunctionalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.inputs=[[0,0],[0xffffffff,1],[5,3],[0x80000000,0x80000000]]
        self.inputs_path=self.root/'inputs.json';functional.write(self.inputs_path,self.inputs)
        self.settings=dict(fuel_per_case=10000,memory_bytes=1048576,execution_timeout_seconds=10)
        self.expected=[(a+b)&0xffffffff for a,b in self.inputs]
    def tearDown(self):self.temp.cleanup()
    def assess(self,source):return functional.assess(source,self.expected,self.inputs_path,self.root,'clang',self.settings)
    def test_unchanged_correct_source_and_wrong_output_are_distinguished(self):
        source=b'unsigned int f(unsigned int a,unsigned int b){return a+b;}'
        result=self.assess(source)
        self.assertEqual(result['status'],'all_cases_match',result)
        self.assertEqual((self.root/'candidate.c').read_bytes(),source)
        self.assertEqual(result['matched_cases'],4)
        result=self.assess(b'unsigned int f(unsigned int a,unsigned int b){return a^b;}')
        self.assertEqual(result['status'],'output_mismatch')
        self.assertLess(result['matched_cases'],4)
    def test_compile_source_contract_and_invalid_encoding_failures(self):
        for source,status in [(b'int f( {','compile_failure'),(b'\xff','invalid_utf8'),(b'','empty_source'),(b'#include "/etc/passwd"','source_contract_rejected'),(b'%:include "/etc/passwd"','source_contract_rejected')]:
            result=self.assess(source);self.assertEqual(result['status'],status)
            self.assertFalse(result['all_cases_match']);self.assertEqual(result['expected_cases'],4)
            self.assertEqual((self.root/'candidate.c').read_bytes(),source)
    def test_nontermination_and_undefined_division_are_failures(self):
        result=self.assess(b'unsigned int f(unsigned int a,unsigned int b){for(;;){} }')
        self.assertEqual(result['status'],'execution_budget_exceeded',result)
        result=self.assess(b'unsigned int f(unsigned int a,unsigned int b){return a/b;}')
        self.assertEqual(result['status'],'execution_error',result)
        self.assertTrue(any(c['status']=='trap' for c in result['execution']['cases']))
    def test_sandbox_rejects_host_imports_wrong_abi_and_memory_overcommit(self):
        import wasmtime
        modules=[('(module (import "env" "host" (func)) (func (export "f") (param i32 i32) (result i32) i32.const 0))','sandbox_rejected'),
                 ('(module (func (export "f") (result i32) i32.const 0))','abi_rejected'),
                 ('(module (memory 32) (func (export "f") (param i32 i32) (result i32) i32.const 0))','executed')]
        for i,(wat,status) in enumerate(modules):
            module=self.root/f'fixture-{i}.wasm';module.write_bytes(wasmtime.wat2wasm(wat))
            result=functional.run_module(module,self.inputs_path,self.root,f'fixture-{i}',self.settings)
            self.assertEqual(result['status'],status,result)
            if i==2:self.assertTrue(all(c['status']=='runtime_error' for c in result['cases']))
    def test_wall_timeout_terminates_process_group(self):
        result=functional.command(['python3','-c','while True: pass'],self.root,'timeout',timeout=0.05,cpu_seconds=1)
        self.assertTrue(result['timed_out']);self.assertNotEqual(result['exit_code'],0)


class DenominatorTests(unittest.TestCase):
    def setUp(self):
        import compare_functional
        self.compare=compare_functional
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.run=self.root/'run';self.frozen=self.root/'frozen';self.run.mkdir();self.frozen.mkdir()
        references={'a':[1,2],'b':[3,4]};functional.write(self.frozen/'references.json',references)
        manifest=dict(targets=[dict(id='a'),dict(id='b')],cases_per_target=2,references_sha256=functional.sha((self.frozen/'references.json').read_bytes()))
        functional.write(self.frozen/'manifest.json',manifest)
        self.rows=[]
        for identifier in ['a','b']:
            folder=self.run/identifier;(folder/'generation').mkdir(parents=True)
            (folder/'generation/completion.bin').write_bytes(b'bad C');(folder/'candidate.c').write_bytes(b'bad C')
            self.rows.append(dict(id=identifier,generated_sha256=functional.sha(b'bad C'),status='compile_failure',compiled=False,matched_cases=0,all_cases_match=False))
        self.write_run()
    def tearDown(self):self.temp.cleanup()
    def write_run(self):
        (self.run/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        functional.write(self.run/'report.json',dict(status='complete',manifest_sha256=functional.sha((self.frozen/'manifest.json').read_bytes()),records_sha256=functional.sha((self.run/'records.jsonl').read_bytes()),completed_targets=len(self.rows),compiled_targets=0,all_case_match_targets=0,matched_cases=0,expected_cases=len(self.rows)*2))
    def test_failed_rows_remain_in_full_denominator_and_missing_rows_fail(self):
        _,counts,_=self.compare.verify(self.run,self.frozen)
        self.assertEqual(counts['targets'],2);self.assertEqual(counts['expected_probe_cases'],4);self.assertEqual(counts['compile_rate'],0)
        self.rows.pop();self.write_run()
        with self.assertRaisesRegex(ValueError,'targets'):self.compare.verify(self.run,self.frozen)
    def test_modified_completion_is_rejected(self):
        (self.run/'a/candidate.c').write_bytes(b'fixed C')
        with self.assertRaisesRegex(ValueError,'modified'):self.compare.verify(self.run,self.frozen)


if __name__=='__main__':unittest.main()
