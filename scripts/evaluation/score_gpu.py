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


def validate_index(index,targets,checkpoint_sha256,max_new_tokens=256):
    expected=[t['id'] for t in targets]
    records=index['records']
    if index['expected_targets']!=len(expected) or [r['id'] for r in records]!=expected:raise ValueError('full ordered validation denominator changed')
    if index['device']!='cuda' or index['max_new_tokens']!=max_new_tokens or index['test_model_examples']!=0:raise ValueError('generation protocol changed')
    for record,target in zip(records,targets):
        if record['prompt_sha256']!=target['prompt_sha256'] or record['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('prompt/checkpoint identity changed')
    return records


def raw_completion(folder,target,checkpoint_sha256,max_new_tokens=256,tokenizer_spec=None):
    report=f.read(folder/'report.json');data=(folder/'completion.bin').read_bytes()
    if report['prompt_sha256']!=target['prompt_sha256'] or report['checkpoint_sha256']!=checkpoint_sha256 or report['max_new_tokens']!=max_new_tokens:raise ValueError('generation report identity changed')
    expected='greedy_eos_and_bpe_tokens_kv_cache_byte_budget' if tokenizer_spec else 'greedy_eos_and_bytes_only_kv_cache'
    if report['backend']!='Cuda { gpu_id: 0 }' or report['decoding']!=expected:raise ValueError('generation backend/decoding changed')
    if tokenizer_spec:
        digest=f.sha(json.dumps(tokenizer_spec,separators=(',',':')).encode())
        if report['tokenizer']!='lexical-byte-bpe-v1' or report['tokenizer_sha256']!=digest or report['max_output_bytes']!=max_new_tokens or len(data)>max_new_tokens:raise ValueError('BPE codec or byte budget changed')
        pieces=[b'']*4+[bytes([b]) for b in range(256)]
        for left,right in tokenizer_spec['merges']:
            if min(left,right)<4 or max(left,right)>=len(pieces):raise ValueError('invalid BPE pieces')
            pieces.append(pieces[left]+pieces[right])
        tokens=report['completion']['tokens'];decoded=b''
        for i,token in enumerate(tokens):
            if token==3 and i==len(tokens)-1:continue
            if not 4<=token<len(pieces):raise ValueError('invalid completion token')
            decoded+=pieces[token]
        if decoded!=data:raise ValueError('BPE token/byte mismatch')
    if data!=bytes(report['completion']['bytes']):raise ValueError('raw completion bytes changed')
    return data


def validate_recovery(recovery,source,selection,selected_sha,state_sha256,manifest,lane,max_new_tokens,generation_model,state):
    if recovery['mode']!='generate' or recovery['optimizer_updates']!=0 or recovery['baseline_step']!=selection['step'] or recovery['baseline_checkpoint_sha256']!=selected_sha or recovery['checkpoint_state_sha256']!=state_sha256:raise ValueError('generation recovery changed selected checkpoint')
    if recovery['corpus_lock_sha256']!=manifest['corpus_lock_sha256'] or recovery['representation']!=lane or recovery['generation_max_new_tokens']!=max_new_tokens or recovery['expected_validation_examples']!=len(manifest['targets']):raise ValueError('generation recovery protocol changed')
    if source['resume/training.safetensors']!=selected_sha or source['resume/state.json']!=recovery['checkpoint_state_sha256'] or generation_model!=state['model']:raise ValueError('generation recovery payload changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--reference-evidence',type=Path,default=Path('evidence/2026-10-20-functional-validation'))
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--generation-results',type=Path,help='Explicit frozen-weight generation recovery; original training/selection remain bound')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=args.results.resolve();e=args.reference_evidence.resolve();corpus=args.corpus.resolve()
    manifest_path=e/'protocol-manifest.json' if (e/'protocol-manifest.json').exists() else e/'manifest.json'
    generation_results=args.generation_results.resolve() if args.generation_results else results
    manifest=f.read(manifest_path);fresh=(results/'fresh-plan.json').exists()
    plan_path=results/('fresh-plan.json' if fresh else 'continuation-plan.json');plan=f.read(plan_path);training=f.read(results/('fresh' if fresh else 'continuation')/'report.json')
    max_new_tokens=manifest['settings']['max_new_tokens']
    if fresh and plan['generation_max_new_tokens']!=max_new_tokens:raise ValueError('fresh generation budget differs from frozen protocol')
    if fresh:
        frozen=manifest['experiment_plan'];improvement=frozen['improvement']
        if plan['steps']!=improvement['additional_updates'] or plan['learning_rate']!=improvement['learning_rate'] or plan['evaluate_every']!=improvement['evaluate_every']:raise ValueError('fresh settings differ from frozen plan')
        if 'model' in frozen and (training['model']!=frozen['model'] or f.read(results/'binary.json')['model']!=frozen['model'] or f.read(results/'source.json')['manifest']['files']['model.json']!=plan['model_sha256']):raise ValueError('model differs from frozen plan')
        expected_decoding='greedy_eos_and_bpe_tokens_kv_cache_byte_budget' if plan.get('tokenizer') else 'greedy_eos_and_bytes_only_kv_cache'
        if manifest['settings']['decoding']!=expected_decoding:raise ValueError('frozen decoder differs from training plan')
        if plan.get('tokenizer') and (plan['tokenizer_sha256']!=f.sha(json.dumps(frozen['tokenizer'],separators=(',',':')).encode()) or plan['max_output_bytes']!=manifest['settings']['max_output_bytes']):raise ValueError('BPE codec differs from frozen plan')

    if f.sha((corpus/'release/lock.json').read_bytes())!=manifest['corpus_lock_sha256'] or training['corpus_lock_sha256']!=manifest['corpus_lock_sha256']:raise ValueError('corpus identity changed')
    if f.sha((corpus/'release/validation.jsonl').read_bytes())!=manifest['validation_split_sha256']:raise ValueError('validation release changed')
    compiler,linker=f.toolchain(manifest['compiler'])
    if f.sha(compiler.read_bytes())!=manifest['compiler_sha256'] or f.sha(linker.read_bytes())!=manifest['linker_sha256'] or importlib.metadata.version('wasmtime')!=manifest['wasmtime_version']:raise ValueError('reference toolchain changed')
    if f.sha(Path(f.__file__).read_bytes())!=manifest['evaluator_sha256'] or f.sha(Path(f.__file__).with_name('wasm_worker.py').read_bytes())!=manifest['worker_sha256']:raise ValueError('bounded evaluator changed')
    for filename,key in [('inputs.json','inputs_sha256'),('references.json','references_sha256')]:
        if f.sha((e/filename).read_bytes())!=manifest[key]:raise ValueError('frozen probes/references changed')
    arms={}
    if fresh:
        if plan['corpus_lock_sha256']!=manifest['corpus_lock_sha256'] or plan['train_examples']!=training['train_examples'] or plan['steps']!=-(-plan['epochs']*training['train_examples']//plan.get('batch_size',1)):raise ValueError('fresh corpus/count plan changed')
        if training['initial_step']!=0 or training['completed_step']!=plan['steps'] or training['test_model_examples']!=0:raise ValueError('fresh training plan changed')
        selection=f.read(results/'selection.json');best=training['best_validation']
        if selection['step']!=best['step'] or selection['nll']!=best['nll'] or selection['checkpoint_sha256']!=best['weights_sha256']:raise ValueError('validation selection changed')
        selected=results/'fresh/checkpoints'/('step-%08d'%selection['step']);state=f.read(selected/'state.json')
        selected_sha=f.sha((selected/'training.safetensors').read_bytes())
        if state['checkpoint_sha256']!=selected_sha or selected_sha!=selection['checkpoint_sha256']:raise ValueError('selected checkpoint integrity mismatch')
        folder=generation_results/'selected-generations';records=validate_index(f.read(folder/'index.json'),manifest['targets'],selected_sha,max_new_tokens)
        lane=manifest['experiment_plan'].get('representation','pcode-plus-ssa-v1')
        if state['representations']!=[lane] or plan.get('representation','pcode-plus-ssa-v1')!=lane:raise ValueError('checkpoint input representation differs from frozen plan')
        if args.generation_results:
            recovery=f.read(generation_results/'generation-plan.json');source=f.read(generation_results/'source.json')['manifest']['files']
            validate_recovery(recovery,source,selection,selected_sha,f.sha((selected/'state.json').read_bytes()),manifest,lane,max_new_tokens,f.read(generation_results/'binary.json')['model'],state)
        spec=state.get('tokenizer_spec')
        if bool(spec)!=bool(plan.get('tokenizer')):raise ValueError('checkpoint codec differs from training plan')
        if spec and (f.sha(json.dumps(spec,separators=(',',':')).encode())!=plan['tokenizer_sha256'] or training['tokenizer_sha256']!=plan['tokenizer_sha256'] or spec['train_split_sha256']!=f.sha((corpus/'release/train.jsonl').read_bytes())):raise ValueError('BPE training provenance changed')
        arms['selected']=dict(folder=folder,records=records,checkpoint_sha256=selected_sha,tokenizer_spec=spec)
    else:
        if training['initial_step']!=plan['baseline_step'] or training['completed_step']!=plan['baseline_step']+plan['additional_steps'] or training['test_model_examples']!=0:raise ValueError('continuation plan changed')
        final=results/'continuation/checkpoints'/Path(training['final_checkpoint']).name
        state=f.read(final/'state.json');final_sha=f.sha((final/'training.safetensors').read_bytes())
        if state['checkpoint_sha256']!=final_sha or state['step']!=training['completed_step']:raise ValueError('final checkpoint integrity mismatch')
        for name,checkpoint_sha in [('baseline',plan['baseline_checkpoint_sha256']),('continued',final_sha)]:
            folder=results/(name+'-generations');records=validate_index(f.read(folder/'index.json'),manifest['targets'],checkpoint_sha,max_new_tokens)
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
                raw=raw_completion(arm['folder']/identifier,target,arm['checkpoint_sha256'],max_new_tokens,arm.get('tokenizer_spec'))
                result=f.assess(raw,references[identifier],e/'inputs.json',destination,manifest['compiler'],manifest['settings'])
            f.write(destination/'result.json',result);total=totals[name];total['targets']+=1;total['compiled']+=result['compiled'];total['all_cases_match']+=result['all_cases_match'];total['matched_cases']+=result['matched_cases'];total['statuses'][result['status']]+=1
            if 'execution' in result:total['targets_reaching_execution']+=1;total['executed_cases']+=len(result['execution'].get('cases',[]))
            exported[name].append(dict(id=identifier,status=result['status'],compiled=result['compiled'],all_cases_match=result['all_cases_match'],matched_cases=result['matched_cases'],raw_base64=base64.b64encode(raw).decode() if raw is not None else None,source_sha256=f.sha(raw) if raw is not None else None))
        if (i+1)%40==0:print('scored',i+1,flush=True)
    for name,total in totals.items():
        total['compile_rate']=total['compiled']/len(manifest['targets']);total['all_case_match_rate']=total['all_cases_match']/len(manifest['targets']);total['case_match_rate']=total['matched_cases']/reference_cases
        (output/(name+'-completions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in exported[name]))
    report=dict(schema_version=1,status='complete',scope='owned_synthetic_validation_not_DecBench',selection=plan['selection'] if fresh else 'fixed_final_step_predeclared_before_training_not_functional_score_search',reference_manifest_sha256=f.sha(manifest_path.read_bytes()),training_plan_sha256=f.sha(plan_path.read_bytes()),generation_driver_sha256=f.read(generation_results/'source.json')['driver_sha256'],scorer_sha256=f.sha(Path(__file__).read_bytes()),checkpoint_sha256={n:a['checkpoint_sha256'] for n,a in arms.items()},reference_comparisons=reference_cases,expected_targets=len(manifest['targets']),arms=totals,test_model_examples=0,raw_output_policy='unchanged_C_no_repairs')
    if args.generation_results:
        if not fresh:raise ValueError('generation recovery requires fresh training selection')
        report['generation_recovery']=dict(plan_sha256=f.sha((generation_results/'generation-plan.json').read_bytes()),optimizer_updates=0,original_training_driver_sha256=f.read(results/'source.json')['driver_sha256'],original_generation_index_sha256=f.sha((results/'selected-generations/index.json').read_bytes()),scope='software_dispatch_fix_same_selected_weights_full_target_retry')
    if not fresh:report['continuation_plan_sha256']=report['training_plan_sha256']
    f.write(output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
