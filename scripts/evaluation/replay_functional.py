"""Strictly replay published raw C outputs and regenerate original owned C.

No checkpoint, Fission clone, or GPU is needed. This replays compilation/runtime
metrics, not model inference. Exact recorded evaluator/compiler/runtime required.
"""
import argparse
import base64
from collections import Counter
import importlib.metadata
from pathlib import Path
import sys
import functional


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();e=args.evidence.resolve();manifest=functional.read(e/'protocol-manifest.json');comparison=functional.read(e/'comparison.json');recipe=functional.read(e/'corpus-config.json')
    if functional.sha((e/'protocol-manifest.json').read_bytes())!=comparison['manifest_sha256']:raise ValueError('protocol changed')
    for filename,field in [('inputs.json','inputs_sha256'),('references.json','references_sha256')]:
        if functional.sha((e/filename).read_bytes())!=manifest[field]:raise ValueError('reference/input changed')
    compiler,linker=functional.toolchain(manifest['compiler'])
    if functional.sha(compiler.read_bytes())!=manifest['compiler_sha256'] or functional.sha(linker.read_bytes())!=manifest['linker_sha256'] or importlib.metadata.version('wasmtime')!=manifest['wasmtime_version']:raise ValueError('recorded toolchain/runtime required')
    if functional.sha(Path(functional.__file__).read_bytes())!=manifest['evaluator_sha256'] or functional.sha(Path(functional.__file__).with_name('wasm_worker.py').read_bytes())!=manifest['worker_sha256']:raise ValueError('recorded evaluator required')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'corpus'));import build
    references=functional.read(e/'references.json');arms={}
    for name,key in [('baseline','baseline-completions.jsonl'),('improved','improved-completions.jsonl')]:
        if functional.sha((e/key).read_bytes())!=comparison['artifacts'][key]:raise ValueError('raw completions changed')
        import json
        rows=[json.loads(s) for s in (e/key).read_text().splitlines()]
        if [r['id'] for r in rows]!=[t['id'] for t in manifest['targets']]:raise ValueError('raw target denominator changed')
        arms[name]={r['id']:r for r in rows}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    totals={name:dict(targets=0,compiled=0,all_cases_match=0,statuses=Counter()) for name in arms};original_cases=0
    for i,target in enumerate(manifest['targets']):
        identifier=target['id'];folder=output/identifier;folder.mkdir()
        family,index=identifier.rsplit('-',1)[0].rsplit('_',1)
        if recipe['families'][family]!='validation':raise ValueError('not a validation source')
        p=build.program(family,int(index),recipe['seed'],'validation');source=p['source'].encode()
        if functional.sha(source)!=target['source_sha256']:raise ValueError('regenerated reference source differs')
        compiled=functional.compile_c(source,folder,'reference',manifest['compiler'])
        if compiled['status']!='compiled':raise ValueError('regenerated reference failed compilation')
        execution=functional.run_module(folder/'reference.wasm',e/'inputs.json',folder,'reference',manifest['settings'])
        if execution['status']!='executed' or [r.get('value') for r in execution['cases']]!=references[identifier]:raise ValueError('regenerated reference execution differs')
        original_cases+=len(execution['cases'])
        for name,rows in arms.items():
            row=rows[identifier];sub=folder/name;sub.mkdir()
            if row['raw_base64'] is None:raise ValueError('this replay requires preserved completion bytes for every target')
            data=base64.b64decode(row['raw_base64'],validate=True)
            if functional.sha(data)!=row['source_sha256']:raise ValueError('raw source hash mismatch')
            result=functional.assess(data,references[identifier],e/'inputs.json',sub,manifest['compiler'],manifest['settings'])
            if result['status']!=row['status']:raise ValueError('replayed failure/success category differs')
            functional.write(sub/'result.json',result);t=totals[name];t['targets']+=1;t['compiled']+=result['compiled'];t['all_cases_match']+=result['all_cases_match'];t['statuses'][result['status']]+=1
        if (i+1)%40==0:print('replayed',i+1,flush=True)
    for name,t in totals.items():
        if any(t[key]!=comparison[name][key] for key in ['targets','compiled','all_cases_match']):raise ValueError('replayed metrics differ')
    report=dict(scope='published_C_and_regenerated_reference_replay_not_new_model_inference',manifest_sha256=comparison['manifest_sha256'],original_comparisons=original_cases,arms=totals,metrics_reproduced=True,test_model_examples=0)
    functional.write(output/'replay-report.json',report);print(__import__('json').dumps(report,indent=2))


if __name__=='__main__':main()
