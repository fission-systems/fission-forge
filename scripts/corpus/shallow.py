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
