"""Derive a label-independent return-u32 input corpus, retaining original evidence.

Reuses the existing audit, including whole-family quarantine and ambiguous-label
rejection. Never copies a frozen release forward without re-finalizing it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

LANE='pcode-plus-ssa-return-u32-v1'
COMPACT='pcode-ssa-return-u32-compact-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--forge',type=Path,required=True);p.add_argument('--lane',choices=[LANE,COMPACT],default=LANE,help='Return-u32 prompt lane; compact drops ABI plumbing');a=p.parse_args();lane=a.lane
    src=a.corpus.resolve();out=a.output.resolve();forge=a.forge.resolve()
    subprocess.run([forge,'corpus-audit','--root',src],check=True,stdout=subprocess.DEVNULL)
    manifest=json.loads((src/'manifest.json').read_text())
    if manifest['representation']!='pcode-plus-ssa-v1':raise ValueError('requires full-SSA source corpus')
    out.mkdir(parents=True,exist_ok=False)
    for name in manifest['artifacts']:
        source=(src/name).resolve()
        if not source.is_relative_to(src) or not source.is_file():raise ValueError('unsafe source artifact')
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    cfg=json.loads((out/'config.json').read_text());cfg['name']+='-return-u32'+('-compact' if lane==COMPACT else '');cfg['representation']=lane;write(out/'config.json',cfg)
    records=[];rows=[]
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        for line in (src/'pool.jsonl').read_text().splitlines():
            row=json.loads(line);pair=row['pair'];old=row['metadata']['sequence_length'];pair['input']['representation']=lane
            write(temp/'raw.json',pair['input']['raw']);write(temp/'profile.json',pair['input']['profile'])
            # Rejection aborts the derived release; unsupported inputs are not hidden.
            result=subprocess.run([forge,'pcode-compile','--raw',temp/'raw.json','--profile',temp/'profile.json','--representation',lane,'--output',temp/'compiled'],capture_output=True,text=True)
            if result.returncode:raise ValueError(pair['id']+': '+result.stderr)
            report=json.loads((temp/'compiled/report.json').read_text());prompt=(temp/'compiled/prompt.txt').read_bytes()
            row['metadata']['prompt_sha256']=report['prompt_sha256'];row['metadata']['sequence_length']=len(prompt)+len(pair['source'].encode())+2
            records.append(dict(id=pair['id'],old_sequence=old,new_sequence=row['metadata']['sequence_length'],prompt_sha256=report['prompt_sha256']))
            rows.append(row);shutil.rmtree(temp/'compiled')
    (out/'pool.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    write(out/'projection.json',dict(schema_version=1,source_corpus_lock_sha256=sha(src/'release/lock.json'),source_manifest_sha256=sha(src/'manifest.json'),representation=lane,uses_labels_for_projection=False,records=records))
    manifest['representation']=lane;manifest['name']=cfg['name'];manifest['config_sha256']=sha(out/'config.json');manifest['artifacts']['projection.json']=sha(out/'projection.json')
    manifest['artifacts']={name:sha(out/name) for name in manifest['artifacts']};write(out/'manifest.json',manifest)
    subprocess.run([forge,'corpus-audit','--root',out,'--finalize'],check=True,stdout=(out/'finalize.json').open('w'))
    print(json.dumps(dict(corpus=str(out),lock_sha256=sha(out/'release/lock.json'),pool_rows=len(rows),max_old_sequence=max(r['old_sequence'] for r in records),max_new_sequence=max(r['new_sequence'] for r in records))))
if __name__=='__main__':main()
