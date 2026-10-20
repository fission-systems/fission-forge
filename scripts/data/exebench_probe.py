"""Inspect a bounded, pinned original ExeBench archive without executing its code.

Report source/provenance/IO feasibility. This is staging, never a train split.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zstandard


def inspect(archive,limit):
    records=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.jsonl.zst'):continue
            with tar.extractfile(member) as binary,zstandard.open(binary,'rt',encoding='utf8') as rows:
                for index in range(limit-len(records)):
                    line=rows.readline(1048577)
                    if not line:break
                    if len(line)>1048576:raise ValueError('source record exceeds 1 MiB staging cap')
                    row=json.loads(line);data=row['text'];source=data.get('func_def','');path=data.get('path','')
                    ref=data.get('ref');signature=data.get('signature')
                    records.append(dict(member=member.name,row_index=index,path=path,project='/'.join(path.split('/')[:2]),upstream_ref=ref,upstream_ref_is_commit=bool(ref and re.fullmatch('[0-9a-f]{40}',ref)),function=data.get('fname'),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_bytes=len(source.encode()),signature=signature,has_real_io=bool(data.get('real_io_pairs')),has_real_dependencies=bool(data.get('real_deps')),contains_floating_types=bool(re.search(r'\b(float|double|_Complex)\b',source)),source_license_metadata=row.get('meta',{}).get('license'),training_approved=False))
            if len(records)>=limit:break
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=100);p.add_argument('--expected-sha256',required=True,help='Independently recorded SHA-256 of the pinned Hub archive');a=p.parse_args()
    if not 1<=a.limit<=1000:raise ValueError('limit must be 1..1000')
    digest=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    if not re.fullmatch('[0-9a-f]{64}',a.expected_sha256) or digest!=a.expected_sha256:raise ValueError('archive digest differs from expected pinned artifact')
    a.output.mkdir(parents=True,exist_ok=False);rows=inspect(a.archive,a.limit)
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    summary=dict(schema_version=1,repo='jordiae/exebench',revision='093085f8558cfd53de8e2c8f4ccc7b9e73dc22ae',revision_binding='operator_download_from_pinned_revision_plus_expected_content_sha256',archive_filename=a.archive.name,archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),sampled_rows=len(rows),projects=dict(Counter(r['project'] for r in rows)),real_io_rows=sum(r['has_real_io'] for r in rows),floating_type_rows=sum(r['contains_floating_types'] for r in rows),explicit_source_license_rows=sum(bool(r['source_license_metadata']) for r in rows),immutable_upstream_commit_rows=sum(r['upstream_ref_is_commit'] for r in rows),external_training_rows_admitted=0,candidate_code_executed=False,limitations=['bounded prefix not representative','original-source licenses still require project/file review','floating point unsupported by current SSA lane','branch refs are not immutable upstream commits','project/benchmark overlap not checked'])
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
