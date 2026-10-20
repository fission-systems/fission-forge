"""Verify complete same-protocol runs and export a before/after audit with raw bytes."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import functional


def verify(root,frozen):
    report=functional.read(root/'report.json');manifest=functional.read(frozen/'manifest.json')
    if report['status']!='complete' or report['manifest_sha256']!=functional.sha((frozen/'manifest.json').read_bytes()):raise ValueError('incomplete or different protocol')
    raw=(root/'records.jsonl').read_bytes()
    if functional.sha(raw)!=report['records_sha256']:raise ValueError('records changed')
    rows=[json.loads(line) for line in raw.splitlines()];ids=[t['id'] for t in manifest['targets']]
    if [r['id'] for r in rows]!=ids or len(set(ids))!=len(ids):raise ValueError('wrong/missing/duplicate targets')
    if functional.sha((frozen/'references.json').read_bytes())!=manifest['references_sha256']:raise ValueError('references changed')
    references=functional.read(frozen/'references.json');completions=[]
    for row in rows:
        identifier=row['id'];data=None
        if 'generated_sha256' in row:
            data=(root/identifier/'generation/completion.bin').read_bytes()
            if functional.sha(data)!=row['generated_sha256'] or data!=(root/identifier/'candidate.c').read_bytes():raise ValueError('completion modified')
        cases=row.get('execution',{}).get('cases',[])
        matched=sum(c['status']=='ok' and c['value']==v for c,v in zip(cases,references[identifier]))
        if matched!=row['matched_cases'] or row['all_cases_match']!=(len(cases)==len(references[identifier]) and matched==len(cases)):raise ValueError('incorrect execution aggregation')
        completions.append(dict(id=identifier,status=row['status'],stop_reason=row.get('stop_reason'),source_sha256=row.get('generated_sha256'),raw_base64=base64.b64encode(data).decode() if data is not None else None))
    totals=dict(targets=len(rows),compiled=sum(r['compiled'] for r in rows),execution_reached=sum('execution' in r for r in rows),executed_probe_cases=sum(len(r.get('execution',{}).get('cases',[])) for r in rows),all_cases_match=sum(r['all_cases_match'] for r in rows),matched_probe_cases=sum(r['matched_cases'] for r in rows),expected_probe_cases=len(rows)*manifest['cases_per_target'],statuses=dict(Counter(r['status'] for r in rows)))
    for actual,key in [('targets','completed_targets'),('compiled','compiled_targets'),('all_cases_match','all_case_match_targets'),('matched_probe_cases','matched_cases'),('expected_probe_cases','expected_cases')]:
        if totals[actual]!=report[key]:raise ValueError('run summary changed')
    totals['compile_rate']=totals['compiled']/totals['targets'];totals['all_case_match_rate']=totals['all_cases_match']/totals['targets']
    return report,totals,completions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['frozen','baseline','improved','training','corpus','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();manifest=functional.read(args.frozen/'manifest.json');plan=manifest['experiment_plan']
    baseline,before,bcompletions=verify(args.baseline,args.frozen);improved,after,icompletions=verify(args.improved,args.frozen)
    if baseline['checkpoint_step']!=plan['baseline']['step'] or improved['checkpoint_step']!=plan['improvement']['final_step'] or baseline['settings']!=improved['settings']:raise ValueError('experiment checkpoints/settings differ')
    training=functional.read(args.training/'report.json');events=[json.loads(s) for s in (args.training/'metrics.jsonl').read_text().splitlines()]
    ids=[i for event in events if event['event']=='train' for i in event['sample_ids']]
    train_ids={p['id'] for p in map(json.loads,(args.corpus/'release/train.jsonl').read_text().splitlines())}
    if len(ids)!=plan['improvement']['additional_updates'] or set(ids)!=train_ids or set(Counter(ids).values())!={3}:raise ValueError('train-only three-epoch coverage mismatch')
    if training['initial_step']!=baseline['checkpoint_step'] or training['completed_step']!=improved['checkpoint_step'] or training['test_model_examples']!=0:raise ValueError('training provenance mismatch')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name,rows in [('baseline-completions.jsonl',bcompletions),('improved-completions.jsonl',icompletions)]:
        (output/name).write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    report=dict(schema_version=1,scope=manifest['scope'],manifest_sha256=baseline['manifest_sha256'],corpus_lock_sha256=manifest['corpus_lock_sha256'],validation_split_sha256=manifest['validation_split_sha256'],evaluation_settings=baseline['settings'],reference_comparisons=manifest['reference_comparisons'],baseline=dict(checkpoint_step=baseline['checkpoint_step'],checkpoint_sha256=baseline['checkpoint_sha256'],**before),improved=dict(checkpoint_step=improved['checkpoint_step'],checkpoint_sha256=improved['checkpoint_sha256'],**after),training=dict(additional_updates=len(ids),unique_train_ids=len(train_ids),visits_per_train_id=3,initial_validation_nll=training['initial_validation']['nll'],final_validation_nll=training['final_validation']['nll'],test_model_examples=0),differences=dict(compiled_targets=after['compiled']-before['compiled'],all_case_match_targets=after['all_cases_match']-before['all_cases_match']),test_model_examples=0,interpretation='finite_64_case_u32_source_equivalence_probe_not_formal_equivalence_or_DecBench',artifacts={name:functional.sha((output/name).read_bytes()) for name in ['baseline-completions.jsonl','improved-completions.jsonl']})
    functional.write(output/'comparison.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
