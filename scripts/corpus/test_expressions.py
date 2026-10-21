"""Composition structure/coverage gates plus unsigned-domain edge cases."""
import json
from pathlib import Path
import unittest
import build
import expressions as e


class ExpressionTests(unittest.TestCase):
    def test_distinct_shapes_and_stratified_primitive_coverage(self):
        cfg=build.config_read(Path('configs/corpus/composition-v1.json'))
        self.assertEqual(len(e.TEMPLATES),96)
        shapes={f:json.dumps(e.shape(n),sort_keys=True) for f,n in e.TEMPLATES.items()}
        self.assertEqual(len(set(shapes.values())),96)
        expected=set().union(*(e.operators(t) for t in e.TEMPLATES.values()))
        for split,count in [('train',72),('validation',12),('test',12)]:
            families=[f for f,s in cfg['families'].items() if s==split]
            self.assertEqual(len(families),count)
            self.assertEqual(sum('select' in e.operators(e.TEMPLATES[f]) for f in families),count//4)
            self.assertEqual(set().union(*(e.operators(e.TEMPLATES[f]) for f in families)),expected)
    def test_wrap_shifts_and_lazy_conditional_oracle(self):
        self.assertEqual(e.evaluate(['add','a',1],0xffffffff,0),0)
        self.assertEqual(e.evaluate(['sub','a',1],0,0),0xffffffff)
        self.assertEqual(e.evaluate(['mul','a','b'],0xffffffff,0xffffffff),1)
        self.assertEqual(e.evaluate(['shr','a',1],0x80000000,0),0x40000000)
        self.assertEqual(e.evaluate(['shl','a',7],0xffffffff,0),0xffffff80)
        self.assertEqual(e.evaluate(['select',['lt','a','b'],1,2],0,0xffffffff),1)
        self.assertEqual(e.evaluate(['select',['lt','a','b'],1,2],0xffffffff,0),2)
    def test_template_variants_change_literals_not_shape(self):
        for family in e.TEMPLATES:
            p=build.program(family,0,314159,'train');q=build.program(family,1,314159,'train')
            self.assertNotEqual(p['source'],q['source'])
            self.assertEqual(p,build.program(family,0,314159,'train'))
            self.assertIn('unsigned int f(unsigned int a, unsigned int b)',p['source'])
            self.assertNotIn('%',p['source'])
            for a,b in build.probes():self.assertTrue(0<=build.oracle(p,a,b)<=0xffffffff)


if __name__=='__main__':unittest.main()
