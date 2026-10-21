"""Owned bounded uint32 expression grammar; separate C renderer/Python oracle.

Templates, not literal variants, are the split unit. No benchmark source is used.
All shifts have literal counts 1..7; unsigned arithmetic wraps after every op.
"""
import hashlib
import json
import random

MASK=2**32-1
BINARY=['add','sub','xor','and','or','mul']
COMMUTATIVE={'add','xor','and','or','mul'}
SYMBOLS=dict(add='+',sub='-',xor='^',and_='&',or_='|',mul='*',shl='<<',shr='>>')
SYMBOLS['and']='&';SYMBOLS['or']='|'


def shape(node):
    if isinstance(node,str):return 'K' if node.startswith('k') else node
    if node[0] in ('shl','shr'):return [node[0],shape(node[1]),'#']
    children=[shape(n) for n in node[1:]]
    if node[0] in COMMUTATIVE:children.sort(key=lambda v:json.dumps(v,sort_keys=True))
    return [node[0],*children]


def templates():
    rng=random.Random(224107);result={};seen=set()
    while len(result)<96:
        leaves=['a','b','k0','k1']
        if rng.random()<0.45:leaves.append(rng.choice(['a','b']))
        rng.shuffle(leaves)
        while len(leaves)>1:
            i=rng.randrange(len(leaves)-1);node=[rng.choice(BINARY),leaves[i],leaves[i+1]]
            if rng.random()<0.12:node=[rng.choice(['shl','shr']),node,1+rng.randrange(7)]
            leaves[i:i+2]=[node]
        node=leaves[0]
        if len(result)%4==0:node=['select',['lt','a','b'],node,['xor','b','k1']]
        key=json.dumps(shape(node),sort_keys=True)
        if key in seen:continue
        seen.add(key);result['expr_%03d'%len(result)]=node
    return result


TEMPLATES=templates()


def instantiate(family,index,seed):
    node=TEMPLATES[family]
    # First literal changes for each adjacent variant, without large decimal labels.
    values={'k0':2+index%14,'k1':2+(index//14+seed%7)%14}
    def walk(n):
        if isinstance(n,str):return values[n] if n.startswith('k') else n
        return [n[0],*[walk(x) if isinstance(x,(list,str)) else x for x in n[1:]]]
    return walk(node)


def render(node):
    if isinstance(node,int):return str(node)+'u'
    if isinstance(node,str):return node
    op=node[0]
    if op=='select':return '('+render(node[1])+' ? '+render(node[2])+' : '+render(node[3])+')'
    symbol='<' if op=='lt' else SYMBOLS[op]
    return '('+render(node[1])+' '+symbol+' '+render(node[2])+')'


def evaluate(node,a,b):
    if isinstance(node,int):return node
    if isinstance(node,str):return {'a':a,'b':b}[node]
    op=node[0]
    if op=='select':return evaluate(node[2] if evaluate(node[1],a,b) else node[3],a,b)
    x,y=evaluate(node[1],a,b),evaluate(node[2],a,b)
    if op=='add':value=x+y
    elif op=='sub':value=x-y
    elif op=='xor':value=x^y
    elif op=='and':value=x&y
    elif op=='or':value=x|y
    elif op=='mul':value=x*y
    elif op=='shl':value=x<<y
    elif op=='shr':value=x>>y
    elif op=='lt':value=int(x<y)
    else:raise ValueError(op)
    return value&MASK


def operators(node):
    if not isinstance(node,list):return set()
    return {node[0]}|set().union(*(operators(child) for child in node[1:]))
