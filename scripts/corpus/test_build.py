"""Generator regressions; optional real-corpus integrity tests use FORGE_CORPUS_FIXTURE."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import build


class GeneratorTests(unittest.TestCase):
    def test_determinism_and_defined_unsigned_domain(self):
        for family in sorted(build.FAMILIES):
            p = build.program(family, 0, 42, 'train')
            self.assertEqual(p, build.program(family, 0, 42, 'train'))
            self.assertNotEqual(p['source'], build.program(family, 1, 42, 'train')['source'])
            self.assertTrue(1 <= p['parameters']['shift'] <= 31)
            self.assertGreater(p['parameters']['m'], 0)
            for a, b in build.probes():
                self.assertTrue(0 <= build.oracle(p, a, b) <= build.MASK)

    def test_ast_removes_names_but_preserves_operations(self):
        def node(name, identifier, literal, opcode='+'):
            return {'kind':'FunctionDecl','id':identifier,'name':name,'inner':[
                {'kind':'ParmVarDecl','id':identifier+'p','name':'renamed','type':{'qualType':'unsigned int'}},
                {'kind':'BinaryOperator','opcode':opcode,'inner':[
                    {'kind':'DeclRefExpr','referencedDecl':{'id':identifier+'p'}},
                    {'kind':'IntegerLiteral','value':literal}, {}]}]}
        a = node('first','0x11','3'); b = node('second','0x22','3')
        self.assertEqual(build.canonical_ast(a), build.canonical_ast(b))
        b = node('third','0x33','8')
        self.assertNotEqual(build.canonical_ast(a), build.canonical_ast(b))
        self.assertEqual(build.canonical_ast(a,True), build.canonical_ast(b,True))
        self.assertNotEqual(build.canonical_ast(a,True), build.canonical_ast(node('x','id','3','^'),True))

    def test_strict_config_and_elf_bounds(self):
        cfg = json.loads(Path('configs/corpus/bootstrap-v1.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            for key, value in [('schema_version', True),('variants_per_family', 0),('optimizations',['O1','O1']),('families',{'affine':'train'})]:
                invalid = copy.deepcopy(cfg); invalid[key] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): build.config_read(path)
        with self.assertRaises((ValueError, build.struct.error)):
            build.function_bytes(b'not an ELF',0,0)


@unittest.skipUnless(os.getenv('FORGE_CORPUS_FIXTURE'), 'set FORGE_CORPUS_FIXTURE to a finalized real builder output')
class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'corpus'
        shutil.copytree(os.environ['FORGE_CORPUS_FIXTURE'], self.root,
                        ignore=shutil.ignore_patterns('attempts'))
        self.forge = str(Path('target/debug/forge').resolve())

    def tearDown(self): self.temp.cleanup()

    def audit(self):
        return subprocess.run([self.forge,'corpus-audit','--root',str(self.root)], capture_output=True,text=True)

    def rehash(self, filename):
        path = self.root/'manifest.json'; manifest=json.loads(path.read_text())
        manifest['artifacts'][filename]=build.sha((self.root/filename).read_bytes())
        path.write_text(json.dumps(manifest))

    def test_frozen_release_and_artifact_tamper(self):
        self.assertEqual(self.audit().returncode,0)
        path=self.root/'release/validation.jsonl'; path.write_bytes(path.read_bytes()+b'\n')
        result=self.audit(); self.assertNotEqual(result.returncode,0)
        self.assertIn('frozen release mismatch',result.stderr)
        path=self.root/'programs.jsonl'; path.write_bytes(path.read_bytes()+b'\n')
        self.assertIn('artifact hash mismatch',self.audit().stderr)

    def test_rehashed_split_swap_and_missing_attempt_are_rejected(self):
        path=self.root/'pool.jsonl'; original=path.read_text(); rows=[json.loads(line) for line in original.splitlines()]
        rows[0]['metadata']['split']='validation'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); self.rehash('pool.jsonl')
        self.assertIn('row assignment mismatch',self.audit().stderr)
        path.write_text(original); self.rehash('pool.jsonl')
        path=self.root/'attempts.jsonl'; path.write_text('\n'.join(path.read_text().splitlines()[1:])+'\n'); self.rehash('attempts.jsonl')
        self.assertIn('missing attempted variants',self.audit().stderr)

    def test_cross_split_clones_refuse_empty_holdout(self):
        path=self.root/'programs.jsonl'
        programs=[json.loads(line) for line in path.read_text().splitlines()]
        reference=next(p for p in programs if p['family']=='affine')
        for p in programs:
            if p['split']=='test':
                p['shape_ast']=reference['shape_ast']; p['shape_sha256']=reference['shape_sha256']
        path.write_text(''.join(json.dumps(p)+'\n' for p in programs)); self.rehash('programs.jsonl')
        path=self.root/'pool.jsonl'; rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row['metadata']['split']=='test': row['metadata']['shape_sha256']=reference['shape_sha256']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); self.rehash('pool.jsonl')
        self.assertIn('empty split',self.audit().stderr)

    def test_unsupported_attempt_stays_in_denominator(self):
        path=self.root/'pool.jsonl'; rows=[json.loads(line) for line in path.read_text().splitlines()]
        removed=rows.pop(0)['metadata']['id']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); self.rehash('pool.jsonl')
        path=self.root/'attempts.jsonl'; attempts=[json.loads(line) for line in path.read_text().splitlines()]
        for a in attempts:
            if a['id']==removed: a.update(status='unsupported',diagnostic='deliberate integrity regression fixture')
        path.write_text(''.join(json.dumps(a)+'\n' for a in attempts)); self.rehash('attempts.jsonl')
        shutil.rmtree(self.root/'release')
        result=subprocess.run([self.forge,'corpus-audit','--root',str(self.root),'--finalize'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['attempted'],len(attempts))
        self.assertEqual(report['unsupported'],1)
        self.assertEqual(report['supported'],len(attempts)-1)
        self.assertEqual(self.audit().returncode,0)

    def test_rehashed_invalid_label_is_rejected(self):
        path=self.root/'pool.jsonl'; rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['pair']['source']='unsigned int f(unsigned int a,unsigned int b){return 0;}'
        rows[0]['pair']['provenance']['source_sha256']=build.sha(rows[0]['pair']['source'].encode())
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); self.rehash('pool.jsonl')
        self.assertIn('row label/profile mismatch',self.audit().stderr)


if __name__ == '__main__': unittest.main()
