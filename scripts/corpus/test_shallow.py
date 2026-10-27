"""Curriculum grammar and split contract; source execution is checked by builder."""
import unittest
import expressions as e
import shallow
import build
from pathlib import Path
import json

class ShallowTests(unittest.TestCase):
    def test_depth_operator_coverage_and_disjoint_template_shapes(self):
        def depth(n):return 0 if not isinstance(n,list) else 1+max(depth(x) for x in n[1:])
        self.assertEqual(len(shallow.TEMPLATES),40)
        seen={}
        for family,node in shallow.TEMPLATES.items():
            self.assertEqual(depth(node),shallow.LEVELS[family])
            key=json.dumps(e.shape(node),sort_keys=True)
            self.assertNotIn(key,seen);seen[key]=family
        for split,count in [('train',16),('validation',16),('test',8)]:
            nodes=[n for f,n in shallow.TEMPLATES.items() if shallow.SPLITS[f]==split]
            self.assertEqual(len(nodes),count)
            self.assertEqual(set().union(*(e.operators(n) for n in nodes)),set(shallow.OPS))
        cfg=build.config_read(Path('configs/corpus/shallow-v1.json'));self.assertEqual(cfg['families'],shallow.SPLITS)
    def test_extra_training_contexts_keep_original_holdout_templates_disjoint(self):
        self.assertEqual(len(shallow.EXTRA_TEMPLATES),8)
        old_shapes={json.dumps(e.shape(n),sort_keys=True) for n in shallow.TEMPLATES.values()}
        for family,node in shallow.EXTRA_TEMPLATES.items():
            self.assertNotIn(json.dumps(e.shape(node),sort_keys=True),old_shapes)
            source=build.program(family,0,314159,'train')['source']
            self.assertIn('return ((b ',source)
        first=build.config_read(Path('configs/corpus/shallow-v1.json'))
        second=build.config_read(Path('configs/corpus/shallow-balanced-v2.json'))
        self.assertEqual({f:s for f,s in first['families'].items() if s!='train'},{f:s for f,s in second['families'].items() if s!='train'})
    def test_literal_variants_unsigned_edges_and_builder_labels(self):
        for family in shallow.TEMPLATES:
            sources=set()
            for i in range(14):
                p=build.program(family,i,314159,shallow.SPLITS[family]);sources.add(p['source'])
                self.assertEqual(p['expression_ast'],shallow.instantiate(family,i,314159))
                for a,b in build.probes():self.assertTrue(0<=build.oracle(p,a,b)<=0xffffffff)
            self.assertEqual(len(sources),14)
        self.assertEqual(e.evaluate(['shl','a',15],0xffffffff,0),0xffff8000)
        self.assertEqual(e.evaluate(['shr','a',2],0x80000000,0),0x20000000)

    def test_wide_variants_are_train_only_distinct_and_leave_holdouts_byte_identical(self):
        base=build.config_read(Path('configs/corpus/shallow-balanced-v2.json'))
        wide=build.config_read(Path('configs/corpus/shallow-wide-v3.json'))
        self.assertEqual({f:s for f,s in base['families'].items() if s!='train'},{f:s for f,s in wide['families'].items() if s!='train'})
        self.assertEqual(set(wide['extra_variants']),{f for f,s in wide['families'].items() if s=='train'})
        for family in wide['extra_variants']:
            sources=set()
            for i in range(14+wide['extra_variants'][family]):
                p=build.program(family,i,wide['seed'],'train');sources.add(p['source'])
                for a,b in build.probes():self.assertTrue(0<=build.oracle(p,a,b)<=0xffffffff)
            self.assertEqual(len(sources),14+wide['extra_variants'][family])
            for i in range(14):
                self.assertEqual(build.program(family,i,wide['seed'],'train')['source'],build.program(family,i,314159,'train')['source'])
        for family in shallow.TEMPLATES:
            if shallow.SPLITS[family]!='train':
                with self.assertRaises(ValueError):shallow.instantiate(family,14,wide['seed'])
        shifts=[shallow.instantiate('shallow_one_'+op+'_a',i,wide['seed'])[2] for op in ('shl','shr') for i in range(14,31)]
        self.assertTrue(all(1<=k<=31 for k in shifts))
    def test_extra_variants_reject_non_train_or_oversized_requests(self):
        import tempfile
        base=json.loads(Path('configs/corpus/shallow-wide-v3.json').read_text())
        for bad in ({'shallow_one_add_b':1},{'shallow_one_add_a':243}):
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'c.json';path.write_text(json.dumps(base|{'extra_variants':bad}))
                with self.assertRaises(ValueError):build.config_read(path)

if __name__=='__main__':unittest.main()
