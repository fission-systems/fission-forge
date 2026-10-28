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

    def test_p0_interpolation_splits_share_one_template_with_disjoint_literals(self):
        cfg=build.config_read(Path('configs/corpus/p0-interpolation-v1.json'))
        self.assertTrue(cfg['diagnostic_shared_shapes'])
        parts=shallow.p0_constants(cfg['seed'])
        self.assertEqual({k:len(v) for k,v in parts.items()},shallow.P0_SIZES)
        everything=[v for vs in parts.values() for v in vs]
        self.assertEqual(len(everything),len(set(everything)));self.assertTrue(all(2<=v<=255 for v in everything))
        for split,family in [(s,f'p0_add_{s}') for s in parts]:
            count=cfg['variants_per_family']+cfg.get('extra_variants',{}).get(family,0)
            self.assertEqual(count,len(parts[split]))
            literals=[shallow.instantiate(family,i,cfg['seed'])[2] for i in range(count)]
            self.assertEqual(literals,parts[split])
            p=build.program(family,0,cfg['seed'],split);self.assertIn(f'a + {parts[split][0]}u',p['source'])
        with self.assertRaises(ValueError):shallow.instantiate('p0_add_test',34,cfg['seed'])

    def test_role_balanced_split_breaks_the_structure_to_letters_confound(self):
        def erase(n):
            if isinstance(n,str):return 'v' if n in ('a','b') else n
            if isinstance(n,int):return 'K'
            return [n[0],*[erase(x) for x in n[1:]]]
        def key(n):return json.dumps(erase(n))
        def roles(n):
            out=[]
            def walk(x):
                if isinstance(x,str) and x in ('a','b'):out.append(x)
                elif isinstance(x,list):[walk(y) for y in x[1:]]
            walk(n);return ''.join(out)
        by={'train':{},'validation':{},'test':{}}
        for name,node in shallow.RB_TEMPLATES.items():
            by[shallow.RB_SPLITS[name]].setdefault(key(node),set()).add(roles(node))
        # every structure in every split occurs with both argument assignments
        for split,structures in by.items():
            for structure,assignments in structures.items():
                self.assertEqual(len(assignments),2,(split,structure,assignments))
        # no structure is shared across splits
        for left,right in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse(set(by[left])&set(by[right]),(left,right))
        # every operator is seen in training, and behavior-equivalent mul/shl share a split
        ops=lambda n:{n[0]} if isinstance(n,list) else set()
        def operators(n):
            return set() if not isinstance(n,list) else {n[0]}.union(*[operators(x) for x in n[1:]])
        train_ops=set().union(*[operators(n) for f,n in shallow.RB_TEMPLATES.items() if shallow.RB_SPLITS[f]=='train'])
        self.assertEqual(train_ops,set(shallow.OPS))
        self.assertEqual(shallow.RB_SPLITS['rb_one_mul_a'],shallow.RB_SPLITS['rb_one_shl_a'])
        cfg=build.config_read(Path('configs/corpus/shallow-rolebal-v2.json'))
        self.assertEqual(cfg['families'],shallow.RB_SPLITS)
        self.assertTrue(set(cfg['extra_variants'])<={f for f,sp in shallow.RB_SPLITS.items() if sp=='train'})
        for family,count in cfg['extra_variants'].items():
            sources={build.program(family,i,cfg['seed'],'train')['source'] for i in range(14+count)}
            self.assertEqual(len(sources),14+count,family)
        with self.assertRaises(ValueError):shallow.instantiate('rb_two_sub_a',14,cfg['seed'])

    def test_role_pairs_are_distinct_functions_so_the_audit_cannot_merge_them(self):
        probes=[(a,b) for a in (0,1,2,7,0xffffffff,0x80000000) for b in (0,1,3,5,0xfffffff0)]
        def value(name,a,b):return e.evaluate(shallow.instantiate(name,0,314159),a,b)
        names={n.rsplit('_',1)[0] for n in shallow.RB_TEMPLATES}
        for stem in names:
            left,right=shallow.RB_TEMPLATES[stem+'_a'],shallow.RB_TEMPLATES[stem+'_b']
            self.assertTrue(any(value(stem+'_a',a,b)!=value(stem+'_b',a,b) for a,b in probes),stem)

if __name__=='__main__':unittest.main()
