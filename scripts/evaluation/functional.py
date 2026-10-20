"""Frozen full-validation generation/compile/execution evaluation; no DecBench score.

The model receives only normalized SSA. References/expected values are used only
by the separate evaluator. Raw completions are never stripped, fixed or renamed.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import sys
import time

SCHEMA=1
FLAGS=['--target=wasm32-unknown-unknown','-std=c11','-O1','-nostdlib','-fno-builtin','-fsanitize=undefined','-fsanitize-trap=all','-fno-sanitize-recover=all','-Werror=return-type','-Werror=implicit-function-declaration','-Wl,--no-entry','-Wl,--export=f','-Wl,--initial-memory=131072','-Wl,--max-memory=1048576']


def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()


def limits(cpu_seconds):
    resource.setrlimit(resource.RLIMIT_CPU,(cpu_seconds,cpu_seconds+1))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16777216,16777216))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def command(cmd,folder,label,timeout=30,cpu_seconds=20):
    """Bounded file-backed logs, kill the entire process group after wall timeout."""
    out=folder/(label+'.stdout');err=folder/(label+'.stderr');start=time.monotonic()
    with out.open('wb') as stdout,err.open('wb') as stderr:
        process=subprocess.Popen(list(map(str,cmd)),stdout=stdout,stderr=stderr,start_new_session=True,
                                 preexec_fn=lambda:limits(cpu_seconds))
        timed_out=False
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    # Even failure output is bounded by RLIMIT_FSIZE. Never put arbitrary logs into prompts.
    return dict(command=list(map(str,cmd)),exit_code=code,timed_out=timed_out,wall_seconds=time.monotonic()-start,
                stdout_sha256=sha(out.read_bytes()),stderr_sha256=sha(err.read_bytes()),
                diagnostic=err.read_text(errors='replace')[-3000:])


def toolchain(cc):
    compiler=Path(shutil.which(str(cc)) or cc).resolve()
    linker=compiler.parent/'wasm-ld'
    if not linker.exists():linker=Path(shutil.which('wasm-ld') or '').absolute()
    if not compiler.is_file() or not linker.is_file():raise ValueError('Clang/wasm-ld required')
    return compiler,linker


def compiler_flags(cc):
    return [*FLAGS,'-fuse-ld='+str(toolchain(cc)[1])]


def compile_c(data,folder,stem,cc):
    source=folder/(stem+'.c');source.write_bytes(data)
    try:text=data.decode('utf-8')
    except UnicodeDecodeError:return dict(status='invalid_utf8',source_sha256=sha(data))
    if not text.strip():return dict(status='empty_source',source_sha256=sha(data))
    # Freestanding scalar C contract: includes/directives/code fences are not repaired.
    # No source-provided include path can read host files during preprocessing.
    checked=text.replace('\\\n','').replace('\\\r\n','')
    if '#' in checked or '%:' in checked or '??' in checked or '_Pragma' in checked or re.search(r'\b(?:asm|__asm|__asm__)\b',checked):
        return dict(status='source_contract_rejected',source_sha256=sha(data),diagnostic='preprocessor directives, pragma and inline assembly are outside the freestanding C contract')
    module=folder/(stem+'.wasm')
    result=command([cc,*compiler_flags(cc),source,'-o',module],folder,stem+'-compile')
    if result['timed_out'] or result['exit_code'] in (-signal.SIGXCPU,-signal.SIGKILL):status='compile_timeout'
    elif result['exit_code']!=0:status='compile_failure'
    elif not module.exists():status='compile_failure'
    else:status='compiled'
    return dict(status=status,source_sha256=sha(data),module_sha256=sha(module.read_bytes()) if module.exists() else None,process=result)


def run_module(module,inputs_path,folder,stem,settings):
    result=command([sys.executable,Path(__file__).with_name('wasm_worker.py'), '--module',module,'--inputs',inputs_path,'--fuel',settings['fuel_per_case'],'--memory-bytes',settings['memory_bytes']],folder,stem+'-execution',timeout=settings['execution_timeout_seconds'],cpu_seconds=6)
    if result['timed_out'] or result['exit_code'] in (-signal.SIGXCPU,-signal.SIGKILL):return dict(status='execution_timeout',process=result,cases=[])
    if result['exit_code']!=0:return dict(status='execution_error',process=result,cases=[])
    try:execution=read(folder/(stem+'-execution.stdout'))
    except (ValueError,OSError):return dict(status='execution_error',process=result,cases=[])
    execution['process']=result;return execution


def assess(data,reference_outputs,inputs_path,folder,cc,settings):
    compiled=compile_c(data,folder,'candidate',cc)
    result=dict(compile=compiled,compiled=compiled['status']=='compiled',expected_cases=len(reference_outputs),matched_cases=0,all_cases_match=False)
    if not result['compiled']:return dict(result,status=compiled['status'])
    execution=run_module(folder/'candidate.wasm',inputs_path,folder,'candidate',settings);result['execution']=execution
    if execution['status']!='executed':return dict(result,status=execution['status'])
    cases=execution['cases']
    if len(cases)!=len(reference_outputs):return dict(result,status='execution_error',diagnostic='missing execution cases')
    result['matched_cases']=sum(case['status']=='ok' and case['value']==expected for case,expected in zip(cases,reference_outputs))
    result['all_cases_match']=result['matched_cases']==len(reference_outputs)
    if result['all_cases_match']:status='all_cases_match'
    elif any(c['status']=='fuel_exhausted' for c in cases):status='execution_budget_exceeded'
    elif any(c['status']!='ok' for c in cases):status='execution_error'
    else:status='output_mismatch'
    return dict(result,status=status)


def freeze(args):
    compiler,linker=toolchain(args.cc)
    args.cc=str(compiler)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    corpus=args.corpus.resolve();forge=args.forge.resolve()
    audit=command([forge,'corpus-audit','--root',corpus],output,'corpus-audit',timeout=120,cpu_seconds=100)
    if audit['exit_code']!=0:raise ValueError('corpus audit failed: '+audit['diagnostic'])
    plan=read(args.plan)
    if plan['schema_version']!=1 or plan['corpus_lock_sha256']!=sha((corpus/'release/lock.json').read_bytes()):raise ValueError('experiment plan/corpus mismatch')
    settings=plan['evaluation']

    rows=[json.loads(line) for line in (corpus/'release/validation.jsonl').read_text().splitlines()]
    programs={p['id']:p for p in map(json.loads,(corpus/'programs.jsonl').read_text().splitlines())}
    inputs=read(corpus/'probes.json');write(output/'inputs.json',inputs)
    references={}
    targets=[]
    for row in rows:
        identifier=row['id']
        if not re.fullmatch(r'[a-z0-9_]+-O[0-3]',identifier):raise ValueError('invalid target identity')
        folder=output/identifier;folder.mkdir()
        write(folder/'raw.json',row['input']['raw']);write(folder/'profile.json',row['input']['profile'])
        prompt=command([forge,'pcode-compile','--raw',folder/'raw.json','--profile',folder/'profile.json','--representation','pcode-plus-ssa-v1','--output',folder/'input'],folder,'prompt')
        if prompt['exit_code']!=0:raise ValueError('invalid frozen validation prompt')
        program_id=identifier.rsplit('-',1)[0];p=programs[program_id]
        if p['split']!='validation' or p['source']!=row['source']:raise ValueError('reference program identity mismatch')
        compiled=compile_c(row['source'].encode(),folder,'reference',args.cc)
        if compiled['status']!='compiled':raise ValueError('reference compilation failed: '+str(compiled))
        execution=run_module(folder/'reference.wasm',output/'inputs.json',folder,'reference',settings)
        expected=p['oracle_outputs']
        if execution['status']!='executed' or [case.get('value') for case in execution['cases']]!=expected:raise ValueError('reference Wasm / frozen native-C oracle disagreement')
        references[identifier]=expected
        targets.append(dict(id=identifier,prompt_sha256=sha((folder/'input/prompt.txt').read_bytes()),source_sha256=sha(row['source'].encode()),reference_module_sha256=compiled['module_sha256'],oracle_sha256=sha(canonical(expected)),family=p['family']))
    write(output/'references.json',references)
    manifest=dict(schema_version=SCHEMA,experiment_plan=plan,experiment_plan_sha256=sha(args.plan.read_bytes()),scope='owned_synthetic_validation_functional_probes_not_DecBench_or_real_project_accuracy',corpus_lock_sha256=sha((corpus/'release/lock.json').read_bytes()),validation_split_sha256=sha((corpus/'release/validation.jsonl').read_bytes()),inputs_sha256=sha((output/'inputs.json').read_bytes()),references_sha256=sha((output/'references.json').read_bytes()),targets=targets,expected_targets=len(rows),cases_per_target=len(inputs),settings=settings,compiler=args.cc,compiler_version=subprocess.check_output([args.cc,'--version'],text=True).splitlines()[0],compiler_flags=compiler_flags(args.cc),compiler_sha256=sha(compiler.read_bytes()),linker=str(linker),linker_sha256=sha(linker.read_bytes()),wasmtime_version=importlib.metadata.version('wasmtime'),forge_binary_sha256=sha(forge.read_bytes()),evaluator_sha256=sha(Path(__file__).read_bytes()),worker_sha256=sha(Path(__file__).with_name('wasm_worker.py').read_bytes()),test_model_examples=0,reference_comparisons=len(rows)*len(inputs),artifact_policy='raw_completions_unmodified_no_source_repair_no_reference_in_model_prompt')
    write(output/'manifest.json',manifest);print(json.dumps(dict(frozen_targets=len(rows),reference_comparisons=manifest['reference_comparisons'],manifest_sha256=sha((output/'manifest.json').read_bytes()))),flush=True)


def evaluate(args):
    frozen=args.frozen.resolve();manifest=read(frozen/'manifest.json');settings=manifest['settings'];forge=args.forge.resolve();checkpoint=args.checkpoint.resolve()
    if manifest['schema_version']!=SCHEMA or manifest['forge_binary_sha256']!=sha(forge.read_bytes()) or manifest['evaluator_sha256']!=sha(Path(__file__).read_bytes()) or manifest['worker_sha256']!=sha(Path(__file__).with_name('wasm_worker.py').read_bytes()):raise ValueError('frozen implementation changed')
    for filename,field in [('inputs.json','inputs_sha256'),('references.json','references_sha256')]:
        if sha((frozen/filename).read_bytes())!=manifest[field]:raise ValueError('frozen input/reference hash mismatch')
    if importlib.metadata.version('wasmtime')!=manifest['wasmtime_version']:raise ValueError('runtime version changed')
    if compiler_flags(manifest['compiler'])!=manifest['compiler_flags'] or sha(Path(manifest['compiler']).read_bytes())!=manifest['compiler_sha256'] or sha(Path(manifest['linker']).read_bytes())!=manifest['linker_sha256']:raise ValueError('compiler/linker changed')
    state=read(checkpoint/'state.json');weights_sha=sha((checkpoint/'training.safetensors').read_bytes())
    if state['schema_version']!=4 or state['corpus']['corpus_lock_sha256']!=manifest['corpus_lock_sha256'] or state['checkpoint_sha256']!=weights_sha:raise ValueError('checkpoint is not from the frozen corpus')
    plan=manifest['experiment_plan']
    if state['step'] not in (plan['baseline']['step'],plan['improvement']['final_step']):raise ValueError('checkpoint step outside predeclared experiment')
    if state['step']==plan['baseline']['step'] and weights_sha!=plan['baseline']['checkpoint_sha256']:raise ValueError('baseline checkpoint differs from predeclared weights')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    references=read(frozen/'references.json');records=[];started=time.monotonic()
    run=dict(schema_version=SCHEMA,status='running',scope=manifest['scope'],manifest_sha256=sha((frozen/'manifest.json').read_bytes()),checkpoint_sha256=weights_sha,checkpoint_state_sha256=sha((checkpoint/'state.json').read_bytes()),checkpoint_step=state['step'],settings=settings,expected_targets=manifest['expected_targets'],cases_per_target=manifest['cases_per_target'],test_model_examples=0)
    write(output/'run.json',run)
    for target in manifest['targets']:
        identifier=target['id'];folder=output/identifier;folder.mkdir();prompt=frozen/identifier/'input/prompt.txt'
        if sha(prompt.read_bytes())!=target['prompt_sha256'] or sha(canonical(references[identifier]))!=target['oracle_sha256']:raise ValueError('frozen target changed')
        generation=command([forge,'generate','--checkpoint',checkpoint,'--prompt',prompt,'--representation','pcode-plus-ssa-v1','--max-new-tokens',settings['max_new_tokens'],'--output',folder/'generation','--device','cpu'],folder,'generation',timeout=settings['generation_timeout_seconds'],cpu_seconds=25)
        record=dict(id=identifier,family=target['family'],generation=generation,compiled=False,matched_cases=0,all_cases_match=False,expected_cases=manifest['cases_per_target'])
        if generation['timed_out'] or generation['exit_code'] in (-signal.SIGXCPU,-signal.SIGKILL):record['status']='generation_timeout'
        elif generation['exit_code']!=0:record['status']='generation_failure'
        else:
            report=read(folder/'generation/report.json');data=(folder/'generation/completion.bin').read_bytes()
            if report['checkpoint_sha256']!=weights_sha or report['prompt_sha256']!=target['prompt_sha256'] or report['decoding']!=settings['decoding']:raise ValueError('generation provenance differs from frozen configuration')
            record['stop_reason']=report['completion']['stop_reason'];record['generated_sha256']=sha(data)
            record.update(assess(data,references[identifier],frozen/'inputs.json',folder,manifest['compiler'],settings))
            if sha((folder/'candidate.c').read_bytes())!=record['generated_sha256']:raise ValueError('candidate bytes changed')
        write(folder/'result.json',record);records.append(record)
        with (output/'records.jsonl').open('a') as log:log.write(json.dumps(record,separators=(',',':'))+'\n')
        if len(records)==1 or len(records)%10==0:print(json.dumps(dict(completed=len(records),total=manifest['expected_targets'],statuses=dict(Counter(r['status'] for r in records)))),flush=True)
    if len(records)!=manifest['expected_targets'] or len({r['id'] for r in records})!=len(records):raise ValueError('incomplete evaluation denominator')
    total=len(records);all_cases=total*manifest['cases_per_target'];compiled=sum(r['compiled'] for r in records);matches=sum(r['all_cases_match'] for r in records);matched_cases=sum(r['matched_cases'] for r in records)
    run.update(status='complete',completed_targets=total,status_counts=dict(Counter(r['status'] for r in records)),stop_reason_counts=dict(Counter(r.get('stop_reason','generation_failed') for r in records)),compiled_targets=compiled,compile_rate=compiled/total,all_case_match_targets=matches,all_case_match_rate=matches/total,expected_cases=all_cases,matched_cases=matched_cases,case_match_rate=matched_cases/all_cases,wall_seconds=time.monotonic()-started,records_sha256=sha((output/'records.jsonl').read_bytes()))
    write(output/'report.json',run);write(output/'run.json',run);print(json.dumps(run,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('--corpus',type=Path,required=True);f.add_argument('--output',type=Path,required=True);f.add_argument('--plan',type=Path,default=Path('configs/evaluation/functional-validation-v1.json'));f.add_argument('--cc',default='clang');f.add_argument('--forge',type=Path,default=Path('target/release/forge'))
    e=commands.add_parser('evaluate');e.add_argument('--frozen',type=Path,required=True);e.add_argument('--checkpoint',type=Path,required=True);e.add_argument('--output',type=Path,required=True);e.add_argument('--forge',type=Path,default=Path('target/release/forge'))
    args=parser.parse_args();freeze(args) if args.command=='freeze' else evaluate(args)


if __name__=='__main__':main()
