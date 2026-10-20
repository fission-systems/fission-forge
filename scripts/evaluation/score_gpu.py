"""Score complete Colab generation batches against frozen owned validation probes.

GPU inference is already recorded. Compilation/runtime reuse the bounded Wasm
contract without changing raw C, repairing output or exposing labels to inference.
"""
import argparse
import base64
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import functional as f


def validate_index(index,targets,checkpoint_sha256):
    expected=[t['id'] for t in targets]
    records=index['records']
    if index['expected_targets']!=len(expected) or [r['id'] for r in records]!=expected:raise ValueError('full ordered validation denominator changed')
    if index['device']!='cuda' or index['max_new_tokens']!=256 or index['test_model_examples']!=0:raise ValueError('generation protocol changed')
    for record,target in zip(records,targets):
        if record['prompt_sha256']!=target['prompt_sha256'] or record['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('prompt/checkpoint identity changed')
    return records


def raw_completion(folder,target,checkpoint_sha256):
    report=f.read(folder/'report.json');data=(folder/'completion.bin').read_bytes()
    if report['prompt_sha256']!=target['prompt_sha256'] or report['checkpoint_sha256']!=checkpoint_sha256 or report['max_new_tokens']!=256:raise ValueError('generation report identity changed')
    if report['backend']!='Cuda { gpu_id: 0 }' or report['decoding']!='greedy_eos_and_bytes_only_kv_cache':raise ValueError('generation backend/decoding changed')
    if data!=bytes(report['completion']['bytes']):raise ValueError('raw completion bytes changed')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--reference-evidence',type=Path,default=Path('evidence/2026-10-20-functional-validation'))
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=args.results.resolve();e=args.reference_evidence.resolve();corpus=args.corpus.resolve()
    manifest=f.read(e/'protocol-manifest.json');plan=f.read(results/'continuation-plan.json');training=f.read(results/'continuation/report.json')
    if f.sha((corpus/'release/lock.json').read_bytes())!=manifest['corpus_lock_sha256'] or training['corpus_lock_sha256']!=manifest['corpus_lock_sha256']:raise ValueError('corpus identity changed')
    if f.sha((corpus/'release/validation.jsonl').read_bytes())!=manifest['validation_split_sha256']:raise ValueError('validation release changed')
    compiler,linker=f.toolchain(manifest['compiler'])
    if f.sha(compiler.read_bytes())!=manifest['compiler_sha256'] or f.sha(linker.read_bytes())!=manifest['linker_sha256'] or importlib.metadata.version('wasmtime')!=manifest['wasmtime_version']:raise ValueError('reference toolchain changed')
    if f.sha(Path(f.__file__).read_bytes())!=manifest['evaluator_sha256'] or f.sha(Path(f.__file__).with_name('wasm_worker.py').read_bytes())!=manifest['worker_sha256']:raise ValueError('bounded evaluator changed')
    for filename,key in [('inputs.json','inputs_sha256'),('references.json','references_sha256')]:
        if f.sha((e/filename).read_bytes())!=manifest[key]:raise ValueError('frozen probes/references changed')
    if training['initial_step']!=plan['baseline_step'] or training['completed_step']!=plan['baseline_step']+plan['additional_steps'] or training['test_model_examples']!=0:raise ValueError('continuation plan changed')
    final=results/'continuation/checkpoints'/Path(training['final_checkpoint']).name
    state=f.read(final/'state.json');final_sha=f.sha((final/'training.safetensors').read_bytes())
    if state['checkpoint_sha256']!=final_sha or state['step']!=training['completed_step']:raise ValueError('final checkpoint integrity mismatch')
    arms={}
    for name,checkpoint_sha in [('baseline',plan['baseline_checkpoint_sha256']),('continued',final_sha)]:
        folder=results/(name+'-generations');records=validate_index(f.read(folder/'index.json'),manifest['targets'],checkpoint_sha)
        arms[name]=dict(folder=folder,records=records,checkpoint_sha256=checkpoint_sha)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    references=f.read(e/'references.json');rows={r['id']:r for r in map(json.loads,(corpus/'release/validation.jsonl').read_text().splitlines())}
    totals={name:dict(targets=0,compiled=0,all_cases_match=0,matched_cases=0,executed_cases=0,targets_reaching_execution=0,statuses=Counter()) for name in arms}
    exported={name:[] for name in arms};reference_cases=0
    for i,target in enumerate(manifest['targets']):
        identifier=target['id'];folder=output/identifier;folder.mkdir();row=rows[identifier]
        if f.sha(row['source'].encode())!=target['source_sha256'] or f.sha(f.canonical(references[identifier]))!=target['oracle_sha256']:raise ValueError('original label changed')
        compiled=f.compile_c(row['source'].encode(),folder,'reference',manifest['compiler'])
        if compiled['status']!='compiled':raise ValueError('original reference no longer compiles')
        execution=f.run_module(folder/'reference.wasm',e/'inputs.json',folder,'reference',manifest['settings'])
        if execution['status']!='executed' or [c.get('value') for c in execution['cases']]!=references[identifier]:raise ValueError('original reference execution differs')
        reference_cases+=len(execution['cases'])
        for name,arm in arms.items():
            record=arm['records'][i];destination=folder/name;destination.mkdir();raw=None
            if record['timed_out'] or record['exit_code']!=0:
                result=dict(status='generation_timeout' if record['timed_out'] else 'generation_failure',compiled=False,all_cases_match=False,matched_cases=0)
            else:
                raw=raw_completion(arm['folder']/identifier,target,arm['checkpoint_sha256'])
                result=f.assess(raw,references[identifier],e/'inputs.json',destination,manifest['compiler'],manifest['settings'])
            f.write(destination/'result.json',result);total=totals[name];total['targets']+=1;total['compiled']+=result['compiled'];total['all_cases_match']+=result['all_cases_match'];total['matched_cases']+=result['matched_cases'];total['statuses'][result['status']]+=1
            if 'execution' in result:total['targets_reaching_execution']+=1;total['executed_cases']+=len(result['execution'].get('cases',[]))
            exported[name].append(dict(id=identifier,status=result['status'],compiled=result['compiled'],all_cases_match=result['all_cases_match'],matched_cases=result['matched_cases'],raw_base64=base64.b64encode(raw).decode() if raw is not None else None,source_sha256=f.sha(raw) if raw is not None else None))
        if (i+1)%40==0:print('scored',i+1,flush=True)
    for name,total in totals.items():
        total['compile_rate']=total['compiled']/len(manifest['targets']);total['all_case_match_rate']=total['all_cases_match']/len(manifest['targets']);total['case_match_rate']=total['matched_cases']/reference_cases
        (output/(name+'-completions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in exported[name]))
    report=dict(schema_version=1,status='complete',scope='owned_synthetic_validation_not_DecBench',selection='fixed_final_step_predeclared_before_training_not_functional_score_search',reference_manifest_sha256=f.sha((e/'protocol-manifest.json').read_bytes()),continuation_plan_sha256=f.sha((results/'continuation-plan.json').read_bytes()),generation_driver_sha256=f.read(results/'source.json')['driver_sha256'],scorer_sha256=f.sha(Path(__file__).read_bytes()),checkpoint_sha256={n:a['checkpoint_sha256'] for n,a in arms.items()},reference_comparisons=reference_cases,expected_targets=len(manifest['targets']),arms=totals,test_model_examples=0,raw_output_policy='unchanged_C_no_repairs')
    f.write(output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
