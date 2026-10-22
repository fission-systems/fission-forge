"""Frozen eight-example training memorization check, not held-out accuracy."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import functional as f


def freeze(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);corpus=a.corpus.resolve();plan=f.read(a.plan)
    rows={r['id']:r for r in map(json.loads,(corpus/'release/train.jsonl').read_text().splitlines())}
    programs={p['id']:p for p in map(json.loads,(corpus/'programs.jsonl').read_text().splitlines())}
    if plan['scope']!='eight_training_examples_memorization_not_heldout_accuracy' or len(plan['ids'])!=8:raise ValueError('invalid train-only plan')
    if f.sha((corpus/'release/train.jsonl').read_bytes())!=plan['train_split_sha256']:raise ValueError('training release changed')
    settings=dict(execution_timeout_seconds=10,fuel_per_case=100000,memory_bytes=1048576)
    (out/'inputs.json').write_bytes((corpus/'probes.json').read_bytes());refs={};targets=[]
    for identifier in plan['ids']:
        row=rows[identifier];folder=out/identifier;folder.mkdir();f.write(folder/'raw.json',row['input']['raw']);f.write(folder/'profile.json',row['input']['profile'])
        result=f.command([a.forge,'pcode-compile','--raw',folder/'raw.json','--profile',folder/'profile.json','--representation',row['input']['representation'],'--output',folder/'ir'],folder,'prompt')
        if result['exit_code']:raise ValueError('invalid train-only prompt')
        compiled=f.compile_c(row['source'].encode(),folder,'reference',str(a.cc.resolve()));expected=programs[identifier.rsplit('-',1)[0]]['oracle_outputs']
        if compiled['status']!='compiled':raise ValueError('reference compilation failed')
        executed=f.run_module(folder/'reference.wasm',out/'inputs.json',folder,'reference',settings)
        if executed['status']!='executed' or [x.get('value') for x in executed['cases']]!=expected:raise ValueError('source oracle mismatch')
        refs[identifier]=expected;targets.append(dict(id=identifier,prompt_sha256=f.sha((folder/'ir/prompt.txt').read_bytes()),source_sha256=f.sha(row['source'].encode()),reference_module_sha256=compiled['module_sha256']))
    if len({tuple(v) for v in refs.values()})!=8:raise ValueError('diagnostic requires eight distinct observed behaviors')
    f.write(out/'references.json',refs);compiler,linker=f.toolchain(a.cc)
    f.write(out/'manifest.json',dict(scope=plan['scope'],plan=plan,targets=targets,settings=settings,corpus_lock_sha256=f.sha((corpus/'release/lock.json').read_bytes()),inputs_sha256=f.sha((out/'inputs.json').read_bytes()),references_sha256=f.sha((out/'references.json').read_bytes()),compiler=str(compiler),compiler_sha256=f.sha(compiler.read_bytes()),linker_sha256=f.sha(linker.read_bytes()),worker_sha256=f.sha(Path(f.__file__).with_name('wasm_worker.py').read_bytes()),evaluator_sha256=f.sha(Path(f.__file__).read_bytes()),reference_comparisons=512,test_model_examples=0,validation_model_examples=0))


def raw_completion(folder,target,weights,cap):
    r=f.read(folder/'report.json');raw=(folder/'completion.bin').read_bytes()
    if r['checkpoint_sha256']!=weights or r['prompt_sha256']!=target['prompt_sha256'] or r['max_new_tokens']!=cap or r['backend']!='Cuda { gpu_id: 0 }' or r['tokenizer']!='byte-v1-260' or len(raw)>cap or bytes(r['completion']['bytes'])!=raw:raise ValueError('train-only raw completion changed')
    tokens=r['completion']['tokens']
    if any(t<4 and (t!=3 or i!=len(tokens)-1) or t>=260 for i,t in enumerate(tokens)) or bytes(t-4 for t in tokens if t>=4)!=raw:raise ValueError('train-only token/byte mismatch')
    return raw


def score(a):
    frozen=a.frozen.resolve();m=f.read(frozen/'manifest.json');results=a.results.resolve();root=results/'syntax-sanity';plan=m['plan'];index=f.read(root/'generations/index.json');cp=root/'checkpoint';state=f.read(cp/'state.json');weights=f.sha((cp/'training.safetensors').read_bytes())
    if f.read(root/'plan.json')!=plan or f.sha((root/'pairs.jsonl').read_bytes())!=plan['dataset_sha256'] or state['dataset_sha256']!=plan['dataset_sha256'] or state['step']!=plan['steps'] or state['model']!=plan['model']:raise ValueError('train-only checkpoint/plan changed')
    if index['checkpoint_sha256']!=weights or index['checkpoint_state_sha256']!=f.sha((cp/'state.json').read_bytes()) or [x['id'] for x in index['records']]!=plan['ids'] or index['expected_targets']!=8:raise ValueError('train-only denominator/checkpoint changed')
    for name,key in [('inputs.json','inputs_sha256'),('references.json','references_sha256')]:
        if f.sha((frozen/name).read_bytes())!=m[key]:raise ValueError('frozen reference changed')
    compiler,linker=f.toolchain(m['compiler'])
    if f.sha(compiler.read_bytes())!=m['compiler_sha256'] or f.sha(linker.read_bytes())!=m['linker_sha256'] or f.sha(Path(f.__file__).read_bytes())!=m['evaluator_sha256'] or f.sha(Path(f.__file__).with_name('wasm_worker.py').read_bytes())!=m['worker_sha256']:raise ValueError('frozen evaluator changed')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);refs=f.read(frozen/'references.json');records=[]
    for target,record in zip(m['targets'],index['records']):
        identifier=target['id'];folder=out/identifier;folder.mkdir();raw=None
        if record['prompt_sha256']!=target['prompt_sha256'] or record['checkpoint_sha256']!=weights:raise ValueError('train-only prompt identity changed')
        if record['exit_code']!=0:result=dict(status='generation_failure',compiled=False,all_cases_match=False,matched_cases=0)
        else:
            raw=raw_completion(root/'generations'/identifier,target,weights,plan['max_new_tokens'])
            result=f.assess(raw,refs[identifier],frozen/'inputs.json',folder,m['compiler'],m['settings'])
        f.write(folder/'result.json',result);records.append(dict(id=identifier,compiled=result['compiled'],all_cases_match=result['all_cases_match'],status=result['status'],matched_cases=result['matched_cases'],raw_base64=base64.b64encode(raw).decode() if raw is not None else None))
    report=dict(scorer_sha256=f.sha(Path(__file__).read_bytes()),scope=m['scope'],examples=8,compiled=sum(r['compiled'] for r in records),all_cases_match=sum(r['all_cases_match'] for r in records),matched_cases=sum(r['matched_cases'] for r in records),total_cases=512,checkpoint_sha256=weights,records=records,test_model_examples=0,validation_model_examples=0)
    f.write(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('freeze');s.add_argument('--corpus',type=Path,required=True);s.add_argument('--plan',type=Path,required=True);s.add_argument('--forge',type=Path,required=True);s.add_argument('--cc',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('score');s.add_argument('--results',type=Path,required=True);s.add_argument('--frozen',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    a=p.parse_args();freeze(a) if a.mode=='freeze' else score(a)
