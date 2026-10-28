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
    nodes=TEMPLATES|EXTRA_TEMPLATES|RB_TEMPLATES|RD_TEMPLATES
    if index>=BASE_VARIANTS and (family in TEMPLATES and SPLITS[family]!='train' or family in RB_SPLITS and RB_SPLITS[family]!='train' or family in RD_SPLITS and RD_SPLITS[family]!='train' or family not in nodes):
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


# Role-balanced split: every training structure appears with BOTH argument assignments, so the
# structure no longer predicts which letter is written and the register must be read. Validation
# and test use unseen operator structures (also with both roles). Validation operators occur in
# training only as inner operators; mul and shl stay in the same split because x*4 and x<<2 are
# the same behavior and the audit quarantines behavior clones across splits.
RB_TEMPLATES={}
RB_SPLITS={}
def _rb(name,node,split):
    RB_TEMPLATES[name]=node;RB_SPLITS[name]=split
def _other(arg):return 'b' if arg=='a' else 'a'
# Pairs are (outer, inner). Each must be ASYMMETRIC in its two arguments: ((x - k) + y) equals
# ((y - k) + x), so a role swap would be the same function (and the audit would drop it as
# ambiguous). Test pairs are unseen in training and validation and also asymmetric.
for split,ones,twos in [('train',['add','and','mul','shl'],[('and','sub'),('add','or'),('mul','xor'),('shl','shr')]),
                        ('validation',['sub','or','xor','shr'],[('sub','and'),('or','add'),('xor','mul'),('shr','shl')]),
                        ('test',[],[('add','xor'),('sub','or'),('or','mul'),('xor','sub')])]:
    for op in ones:
        for arg in ('a','b'):_rb(f'rb_one_{op}_{arg}',[op,arg,'k0'],split)
    for outer,inner in twos:
        for arg in ('a','b'):
            _rb(f'rb_two_{outer}_{arg}' if split!='test' else f'rb_test_{outer}_{arg}',[outer,[inner,arg,'k0'],3 if outer in ('shl','shr') else _other(arg)],split)


# Operator-pair diversity: almost every (outer, inner) pair is trained, with both argument roles; a
# few whole pairs are held out for validation and test, so a held-out structure is a NEW combination of
# operators that each occur in training as both an outer and an inner operator. Rules:
#  * a pair whose role swap is the same function (for example (x - k) + y) is excluded: the audit drops
#    it as an ambiguous prompt and it could not teach the register-to-letter binding;
#  * mul and shl form one class and always share a split (x*4 and x<<2 are the same behavior, and the
#    audit quarantines behavior clones across splits);
#  * every operator is an outer and an inner operator of at least two training pairs.
import random as _random
RD_CLASS={'add':'add','sub':'sub','and':'and','or':'or','xor':'xor','mul':'M','shl':'M','shr':'shr'}
RD_SEED=271828
def _rd_symmetric(outer,inner):
    """True when the two role assignments are indistinguishable at k=5: a role swap that leaves the
    function unchanged (for example (x - k) + y) or a degenerate one (for example (x & k) >> 3 is always 0)."""
    probes=[(a,b) for a in (0,1,2,7,0xffffffff,0x80000000) for b in (0,1,3,5,0xfffffff0)]
    def node(arg):return [outer,[inner,arg,5],3 if outer in ('shl','shr') else _other(arg)]
    return all(e.evaluate(node('a'),a,b)==e.evaluate(node('b'),a,b) for a,b in probes)
def _rd_absorbed(outer,inner):
    """(x | k) >> 3 and (x ^ k) >> 3 equal x >> 3 for k < 8: the shift discards what the inner operator
    changed, so the pair is a behavior clone of a different structure and would be quarantined."""
    return outer=='shr' and inner in ('or','xor','and')
def rd_pairs():
    return [(o,i) for o in OPS for i in OPS if o!=i and not _rd_symmetric(o,i) and not _rd_absorbed(o,i)]
def rd_split_pairs():
    groups={}
    for o,i in rd_pairs():groups.setdefault((RD_CLASS[o],RD_CLASS[i]),[]).append((o,i))
    keys=sorted(groups);_random.Random(RD_SEED).shuffle(keys)
    held={'validation':[],'test':[]};count={'validation':0,'test':0}
    def covered(taken):
        train=[p for k in keys if k not in taken for p in groups[k]]
        return all(sum(1 for o,_ in train if o==op)>=2 and sum(1 for _,i in train if i==op)>=2 for op in OPS)
    taken=set()
    for k in keys:
        for split in ('validation','test'):
            if count[split]<8 and covered(taken|{k}):
                taken.add(k);held[split].append(k);count[split]+=len(groups[k]);break
    out={'train':[],'validation':[],'test':[]}
    for k in keys:
        split='validation' if k in held['validation'] else 'test' if k in held['test'] else 'train'
        out[split].extend(groups[k])
    return {k:sorted(v) for k,v in out.items()}
RD_TEMPLATES={}
RD_SPLITS={}
def _rd(name,node,split):
    RD_TEMPLATES[name]=node;RD_SPLITS[name]=split
for _split,_pairs in rd_split_pairs().items():
    for _o,_i in _pairs:
        for _arg in ('a','b'):
            _rd(f'rd_two_{_o}_{_i}_{_arg}' if _split=='train' else f'rd_{_split[:3]}_{_o}_{_i}_{_arg}',
                [_o,[_i,_arg,'k0'],3 if _o in ('shl','shr') else _other(_arg)],_split)
for _op in OPS:
    for _arg in ('a','b'):_rd(f'rd_one_{_op}_{_arg}',[_op,_arg,'k0'],'train')
