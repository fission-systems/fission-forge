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

if __name__=='__main__':unittest.main()
