"""Owned one/two-operation curriculum; template-disjoint operand-role holdouts.

Only unsigned arithmetic/bitwise operations and bounded literal shifts. No source
from evaluation benchmarks or external datasets. Parameters vary only in train-
independent deterministic literal slots; a complete template owns one split.
"""
import hashlib
import expressions as e

OPS=['add','sub','mul','and','or','xor','shl','shr']
TEMPLATES={}
SPLITS={}
LEVELS={}
def add(name,node,split,level):
    TEMPLATES[name]=node;SPLITS[name]=split;LEVELS[name]=level
for op in OPS:
    add('shallow_one_'+op+'_a',[op,'a','k0'],'train',1)
    add('shallow_one_'+op+'_b',[op,'b','k0'],'validation',1)
    # Novel two-op structure, not a literal variant of any training template.
    add('shallow_test_'+op,[op,['add','a','b'],'k0'] if op in ('shl','shr') else [op,[op,'a','b'],'k0'],'test',2)
    inner='xor' if op in ('add','sub','mul','shl','shr') else 'add'
    for arg,other,split in [('a','b','train'),('b','a','validation')]:
        node=[op,[inner,arg,'k0'],3 if op in ('shl','shr') else other]
        add('shallow_two_'+op+'_'+arg,node,split,2)

BASE_VARIANTS=14
BASE_CONSTANTS=range(2,2+BASE_VARIANTS)
def wide_constant(family,index,seed,shift):
    """Distinct deterministic train-only literals outside the base 2..15; shifts stay below 32."""
    pool=range(1,32) if shift else range(2,256)
    order=sorted((v for v in pool if v not in BASE_CONSTANTS),key=lambda v:hashlib.sha256(f'{seed}:{family}:{v}'.encode()).digest())
    if index-BASE_VARIANTS>=len(order):raise ValueError('wide literal pool exhausted')
    return order[index-BASE_VARIANTS]

def instantiate(family,index,seed):
    if family in P0_TEMPLATES:return instantiate_p0(family,index,seed)
    nodes=TEMPLATES|EXTRA_TEMPLATES
    if index>=BASE_VARIANTS and (family in TEMPLATES and SPLITS[family]!='train' or family not in nodes):
        raise ValueError('wide literal variants are train-only')
    def walk(n,parent=None):
        if isinstance(n,str):
            if n!='k0':return n
            return 2+index%BASE_VARIANTS if index<BASE_VARIANTS else wide_constant(family,index,seed,parent in ('shl','shr'))
        return [n[0],*[walk(x,n[0]) if isinstance(x,(list,str)) else x for x in n[1:]]]
    return walk(nodes[family])

# Additional training contexts cover body-first b without admitting any original
# validation/test template. Existing 40 template definitions/splits stay intact.
EXTRA_TEMPLATES={}
for op in OPS:
    inner={'add':'and','sub':'or','mul':'or','and':'xor','or':'xor','xor':'and','shl':'add','shr':'add'}[op]
    EXTRA_TEMPLATES['shallow_extra_'+op+'_b']=[op,[inner,'b','k0'],3 if op in ('shl','shr') else 'a']


# P0 constant-interpolation diagnostic: ONE template (a + k) in all three splits with
# disjoint literals drawn from a seeded partition of 2..255. Only valid with a corpus
# configuration that sets `diagnostic_shared_shapes`; it is never a headline benchmark.
P0_TEMPLATES={f'p0_add_{split}':['add','a','k0'] for split in ('train','validation','test')}
P0_SIZES={'train':180,'validation':34,'test':34}
def p0_constants(seed):
    order=sorted(range(2,256),key=lambda v:hashlib.sha256(f'p0:{seed}:{v}'.encode()).digest())
    start=0;out={}
    for split in ('train','validation','test'):
        out[split]=order[start:start+P0_SIZES[split]];start+=P0_SIZES[split]
    return out

def instantiate_p0(family,index,seed):
    split=family.rsplit('_',1)[1]
    constants=p0_constants(seed)[split]
    if not 0<=index<len(constants):raise ValueError('P0 variant index out of range')
    return ['add','a',constants[index]]
