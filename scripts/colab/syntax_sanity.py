"""Train-only conditional memorization diagnostic, never a generalization score."""
import hashlib
import json
from pathlib import Path
import subprocess

OPS=['add','sub','mul','and','or','xor','shl','shr']
def canonical_rows(rows):return (''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows)).encode()
def select(corpus):
    training=[json.loads(l) for l in (corpus/'release/train.jsonl').read_text().splitlines()]
    rows=[]
    for op in OPS:
        candidates=[r for r in training if r['id'].startswith('shallow_one_'+op+'_a_')]
        if not candidates:raise ValueError('train-only diagnostic lacks '+op)
        rows.append(min(candidates,key=lambda r:r['id']))
    if len({r['source'] for r in rows})!=8:raise ValueError('diagnostic labels must differ')
    return rows

def prepare(corpus):
    rows=select(corpus)
    return dict(scope='eight_training_examples_memorization_not_heldout_accuracy',ids=[r['id'] for r in rows],dataset_sha256=hashlib.sha256(canonical_rows(rows)).hexdigest(),train_split_sha256=hashlib.sha256((corpus/'release/train.jsonl').read_bytes()).hexdigest(),steps=600,learning_rate=0.003,seed=42,max_new_tokens=128,model=dict(vocab_size=260,hidden_size=128,heads=4,layers=2,feed_forward_size=512,max_seq_len=2048,rotary_positions=True),test_model_examples=0,validation_model_examples=0,selection='fixed_final_step_before_functional_scoring')

def run(plan,result,forge,corpus,env,measure,write,sha):
    rows=select(corpus);folder=result/'syntax-sanity';folder.mkdir()
    if prepare(corpus)!=plan:raise ValueError('frozen train-only plan changed')
    dataset=folder/'pairs.jsonl';dataset.write_bytes(canonical_rows(rows));config=folder/'model.json';write(config,plan['model']);write(folder/'plan.json',plan)
    checkpoint=folder/'checkpoint'
    report=measure([forge,'train-smoke','--config',str(config),'--dataset',str(dataset),'--output',str(checkpoint),'--steps',str(plan['steps']),'--learning-rate',str(plan['learning_rate']),'--seed',str(plan['seed']),'--device','cuda'],'syntax-sanity-training',env,timeout=240)
    weight_sha=sha(checkpoint/'training.safetensors');state_sha=sha(checkpoint/'state.json')
    output=folder/'generations';output.mkdir();records=[]
    for pair in rows:
        prompt=folder/pair['id'];prompt.mkdir();write(prompt/'raw.json',pair['input']['raw']);write(prompt/'profile.json',pair['input']['profile'])
        subprocess.run([forge,'pcode-compile','--raw',str(prompt/'raw.json'),'--profile',str(prompt/'profile.json'),'--representation',pair['input']['representation'],'--output',str(prompt/'ir')],env=env,check=True,stdout=subprocess.DEVNULL,timeout=30)
        p=prompt/'ir/prompt.txt'
        generated=subprocess.run([forge,'generate','--checkpoint',str(checkpoint),'--prompt',str(p),'--representation',pair['input']['representation'],'--output',str(output/pair['id']),'--max-new-tokens',str(plan['max_new_tokens']),'--device','cuda'],env=env,capture_output=True,text=True,timeout=30)
        records.append(dict(id=pair['id'],exit_code=generated.returncode,diagnostic=generated.stderr[-2000:],prompt_sha256=sha(p),checkpoint_sha256=weight_sha))
    if weight_sha!=sha(checkpoint/'training.safetensors') or state_sha!=sha(checkpoint/'state.json'):raise ValueError('generation modified train-only checkpoint')
    write(output/'index.json',dict(scope=plan['scope'],records=records,expected_targets=8,checkpoint_sha256=weight_sha,checkpoint_state_sha256=state_sha,test_model_examples=0,validation_model_examples=0))
