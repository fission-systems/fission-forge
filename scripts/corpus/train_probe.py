"""Bounded CPU acceptance run on a frozen corpus; never reads test rows into the model.

Checks two updates plus resume against three uninterrupted updates, rejects changed
sampler/weights, then exercises the longest training row and full validation.
This is software/long-context evidence, not model-quality evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')


def order(seed,count,epoch=0):
    state=int(hashlib.sha256(f'forge-fisher-yates-v1:{seed}:{epoch}'.encode()).hexdigest()[:16],16) or 1
    result=list(range(count)); mask=2**64-1
    for i in range(count-1,0,-1):
        state ^= (state << 13) & mask; state ^= state >> 7; state ^= (state << 17) & mask
        j=state % (i+1); result[i],result[j]=result[j],result[i]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--forge',type=Path,default=Path('target/release/forge'))
    p.add_argument('--config',type=Path,default=Path('configs/model/tiny-corpus-8k.json'))
    p.add_argument('--skip-full-validation',action='store_true',help='Skip longest-train/full-validation resource measurement')
    args=p.parse_args();root=args.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    forge=args.forge.resolve(); corpus=args.corpus.resolve(); config=args.config.resolve()
    results=[]
    def command(name,cmd,success=True):
        start=time.monotonic()
        # Darwin time -l records peak resident bytes for the actual CLI process.
        actual=['/usr/bin/time','-l',*map(str,cmd)] if Path('/usr/bin/time').exists() and __import__('sys').platform=='darwin' else list(map(str,cmd))
        result=subprocess.run(actual,capture_output=True,text=True,timeout=900)
        (root/(name+'.stdout')).write_text(result.stdout);(root/(name+'.stderr')).write_text(result.stderr)
        match=re.search(r'(\d+)\s+maximum resident set size',result.stderr)
        record=dict(name=name,command=list(map(str,cmd)),exit_code=result.returncode,wall_seconds=time.monotonic()-start,maximum_resident_bytes=int(match[1]) if match else None)
        results.append(record);print(json.dumps(record),flush=True)
        if (result.returncode==0)!=success: raise RuntimeError(result.stderr)
        return result
    def train(name,steps,resume=None,seed=42,limit=2):
        cmd=[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',root/name,'--steps',steps,'--seed',seed,'--learning-rate','0.001','--attention-chunk','256','--validation-limit',limit,'--evaluate-every','1','--checkpoint-every','1']
        if resume: cmd+=['--resume',resume]
        command(name,cmd); return read(root/name/'report.json')
    first=train('first',2); resumed=train('resumed',1,first['final_checkpoint']); full=train('uninterrupted',3)
    a=Path(resumed['final_checkpoint']);b=Path(full['final_checkpoint'])
    assert sha(a/'training.safetensors')==sha(b/'training.safetensors'), 'resumed weights/Adam moments differ'
    assert resumed['sampler']==full['sampler']
    def training_ids(name):
        return [event['sample_ids'] for event in map(json.loads,(root/name/'metrics.jsonl').read_text().splitlines()) if event['event']=='train']
    assert training_ids('first')+training_ids('resumed')==training_ids('uninterrupted')
    assert resumed['final_validation']['nll']==full['final_validation']['nll']
    rejected=command('changed-settings',[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',root/'rejected-settings','--steps','1','--resume',a,'--learning-rate','0.001','--validation-limit','1','--evaluate-every','1','--checkpoint-every','1'],False)
    assert 'settings changed' in rejected.stderr and not (root/'rejected-settings').exists()
    import shutil
    corrupted=root/'corrupted-checkpoint';shutil.copytree(a,corrupted)
    weights=corrupted/'training.safetensors'; data=bytearray(weights.read_bytes());data[-1]^=1;weights.write_bytes(data)
    rejected=command('changed-weights',[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',root/'rejected-weights','--steps','1','--resume',corrupted,'--learning-rate','0.001','--validation-limit','2','--evaluate-every','1','--checkpoint-every','1'],False)
    assert 'weights digest mismatch' in rejected.stderr and not (root/'rejected-weights').exists()
    longest=None
    if not args.skip_full_validation:
        rows=[json.loads(line) for line in (corpus/'release/train.jsonl').read_text().splitlines()]
        metadata={row['metadata']['id']:row['metadata'] for row in map(json.loads,(corpus/'pool.jsonl').read_text().splitlines())}
        maximum=max(metadata[row['id']]['sequence_length'] for row in rows)
        targets={i for i,row in enumerate(rows) if metadata[row['id']]['sequence_length']==maximum}
        seed=next(seed for seed in range(1,100001) if order(seed,len(rows))[0] in targets)
        longest=train('longest-full-validation',1,seed=seed,limit=0)
        assert longest['max_training_sequence']==maximum
        assert longest['final_validation']['selection']=='full_split'
        assert longest['final_validation']['examples']==longest['validation_examples']
    bad_sampler=root/'bad-sampler-checkpoint';shutil.copytree(a,bad_sampler)
    state=read(bad_sampler/'state.json');state['corpus']['sampler']['cursor']+=1;write(bad_sampler/'state.json',state)
    rejected=command('changed-sampler',[forge,'train-corpus','--config',config,'--corpus',corpus,'--output',root/'rejected-sampler','--steps','1','--resume',bad_sampler,'--learning-rate','0.001','--validation-limit','2','--evaluate-every','1','--checkpoint-every','1'],False)
    assert 'sampler position' in rejected.stderr and not (root/'rejected-sampler').exists()
    # Exercise schema-4 inference without making any C-quality claim.
    sample=json.loads((corpus/'release/train.jsonl').read_text().splitlines()[0]);write(root/'generation-raw.json',sample['input']['raw']);write(root/'generation-profile.json',sample['input']['profile'])
    command('generation-prompt',[forge,'pcode-compile','--raw',root/'generation-raw.json','--profile',root/'generation-profile.json','--representation','pcode-plus-ssa-v1','--output',root/'generation-input'])
    command('generation',[forge,'generate','--checkpoint',a,'--prompt',root/'generation-input/prompt.txt','--representation','pcode-plus-ssa-v1','--max-new-tokens','4','--output',root/'generation'])
    rejected=command('changed-generation-weights',[forge,'generate','--checkpoint',corrupted,'--prompt',root/'generation-input/prompt.txt','--representation','pcode-plus-ssa-v1','--max-new-tokens','4','--output',root/'rejected-generation'],False)
    assert 'integrity mismatch' in rejected.stderr and not (root/'rejected-generation').exists()
    summary=dict(scope='software_resume_validation_and_bounded_long_context_CPU_probe_not_model_quality',forge_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),forge_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),forge_binary_sha256=sha(forge),config=read(config),corpus_lock_sha256=sha(corpus/'release/lock.json'),weights_and_adam_resume_byte_identical=True,sample_order_resume_identical=True,validation_resume_identical=True,changed_settings_rejected=True,corrupted_weights_rejected=True,corrupted_sampler_rejected=True,corrupted_generation_weights_rejected=True,longest_train_sequence=longest['max_training_sequence'] if longest else None,full_validation_examples=longest['final_validation']['examples'] if longest else None,test_model_examples=0,runs=results)
    write(root/'verification.json',summary)


if __name__=='__main__': main()
