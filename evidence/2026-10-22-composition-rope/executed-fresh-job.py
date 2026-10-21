"""Fresh-model composition corpus experiment; invoked by the bounded CUDA driver."""
import json
from pathlib import Path
import subprocess
import time


def first_index(seed,count):
    import hashlib
    order=list(range(count));state=int(hashlib.sha256(f'forge-fisher-yates-v1:{seed}:0'.encode()).hexdigest()[:16],16) or 1
    mask=2**64-1
    for i in range(count-1,0,-1):
        state^=(state<<13)&mask;state^=state>>7;state^=(state<<17)&mask
        j=state%(i+1);order[i],order[j]=order[j],order[i]
    return order[0]


def run(job,root,result,forge,config,corpus,env,train,write,sha,status,start,total_mib):
    write(result/'fresh-plan.json',job)
    training=[json.loads(line) for line in (corpus/'release/train.jsonl').read_text().splitlines()]
    sizes={json.loads(line)['metadata']['id']:json.loads(line)['metadata']['sequence_length'] for line in (corpus/'pool.jsonl').read_text().splitlines()}
    if job.get('tokenizer'):
        if sha(root/'tokenizer.json')!=job['tokenizer_file_sha256']:raise RuntimeError('BPE payload changed')
        encoded=json.loads(subprocess.check_output([forge,'tokenizer-stats','--corpus',str(corpus),'--tokenizer',str(root/'tokenizer.json')],env=env,text=True,timeout=90))
        if encoded['tokenizer_sha256']!=job['tokenizer_sha256']:raise RuntimeError('BPE identity changed')
        write(result/'tokenizer-stats.json',encoded);sizes={r['id']:r['sequence_length'] for r in encoded['records']}
    longest=max(range(len(training)),key=lambda i:sizes[training[i]['id']])
    seed=next(seed for seed in range(1,100001) if first_index(seed,len(training))==longest)
    probe=train('fresh-preflight',1,seed=seed,limit=2,learning_rate=job['learning_rate'])
    events=[json.loads(line) for line in (result/'fresh-preflight/metrics.jsonl').read_text().splitlines()]
    if next(e for e in events if e['event']=='train')['sample_ids']!=[training[longest]['id']]:raise RuntimeError('Python/Rust sampler disagreement')
    if probe['max_training_sequence']!=sizes[training[longest]['id']]:raise RuntimeError('longest input preflight mismatch')
    resources=json.loads((result/'fresh-preflight-resources.json').read_text())
    if resources['observed_whole_device_peak_mib']>total_mib*0.85:raise RuntimeError('longest input exceeds safe GPU memory budget')
    write(result/'fresh-preflight-verification.json',dict(seed=seed,longest_train_id=training[longest]['id'],sequence_length=probe['max_training_sequence'],model_parameters=probe['parameters'],sampler_first_index_verified=True))
    report=train('fresh',job['steps'],seed=42,limit=0,every=job['evaluate_every'],timeout=900,learning_rate=job['learning_rate'])
    selected=Path(report['best_validation']['checkpoint'])
    write(result/'selection.json',dict(policy='lowest_full_validation_NLL_among_saved_checkpoints_before_functional_scoring',step=report['best_validation']['step'],nll=report['best_validation']['nll'],checkpoint_sha256=sha(selected/'training.safetensors'),checkpoint= str(selected),final_checkpoint=report['final_checkpoint'],test_model_examples=0))
    pairs=[json.loads(line) for line in (corpus/'release/validation.jsonl').read_text().splitlines()]
    prompts=result/'validation-prompts';prompts.mkdir();output=result/'selected-generations';output.mkdir();records=[];checkpoint_sha=sha(selected/'training.safetensors')
    for i,pair in enumerate(pairs):
        if time.monotonic()-start>1500:raise TimeoutError('fresh generation global wall budget exceeded')
        status('selected-generations',completed=i,total=len(pairs));folder=prompts/pair['id'];folder.mkdir()
        write(folder/'raw.json',pair['input']['raw']);write(folder/'profile.json',pair['input']['profile'])
        subprocess.run([forge,'pcode-compile','--raw',str(folder/'raw.json'),'--profile',str(folder/'profile.json'),'--representation','pcode-plus-ssa-v1','--output',str(folder/'input')],env=env,check=True,stdout=subprocess.DEVNULL,timeout=30)
        prompt=folder/'input/prompt.txt'
        command=[forge,'generate','--checkpoint',str(selected),'--prompt',str(prompt),'--representation','pcode-plus-ssa-v1','--max-new-tokens',str(job['generation_max_new_tokens']),'--output',str(output/pair['id']),'--device','cuda']
        try:
            generated=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
            record=dict(id=pair['id'],exit_code=generated.returncode,timed_out=False,diagnostic=generated.stderr[-2000:],prompt_sha256=sha(prompt),checkpoint_sha256=checkpoint_sha)
        except subprocess.TimeoutExpired:record=dict(id=pair['id'],exit_code=None,timed_out=True,prompt_sha256=sha(prompt),checkpoint_sha256=checkpoint_sha)
        records.append(record)
    write(output/'index.json',dict(records=records,expected_targets=len(pairs),max_new_tokens=job['generation_max_new_tokens'],device='cuda',test_model_examples=0))
