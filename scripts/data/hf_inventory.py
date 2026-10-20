"""Read-only pinned HF cards/metadata plus bounded Viewer schema previews.

Viewer serves HEAD, so previews are explicitly not training-admissible pinned
rows. No remote dataset scripts, wrappers, binaries, or large archives execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def fetch(url,json_body=True):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Fission-Forge-dataset-research/1'}),timeout=30) as r:
            data=r.read(8388609)
        if len(data)>8388608:raise ValueError('metadata exceeds 8 MiB cap')
        return json.loads(data) if json_body else data.decode()
    except (OSError,ValueError) as e:return dict(error=str(e),url=url)


def inspect(source,output):
    repo=source['repo'];revision=source['revision'];folder=output/repo.replace('/','--');folder.mkdir()
    info=fetch(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}')
    card=fetch(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md',False)
    if isinstance(card,str):(folder/'README.upstream.md').write_text(card)
    reserved=source['policy']=='evaluation_reserved'
    result=dict(**source,pinned_revision_returned=info.get('sha'),pinned_metadata_verified=info.get('sha')==revision,metadata_error=info.get('error'),card_metadata=info.get('cardData'),files=[s['rfilename'] for s in info.get('siblings',[])],card_sha256=hashlib.sha256(card.encode()).hexdigest() if isinstance(card,str) else None,training_approved=False,benchmark_overlap='not_measured',source_license_verified=False)
    if reserved:result.update(content_sampled=False,reason_content_not_sampled='evaluation reservation')
    else:
        params=urllib.parse.urlencode({'dataset':repo});splits=fetch('https://datasets-server.huggingface.co/splits?'+params);result['viewer_splits']=splits
        if splits.get('splits'):
            train=next((s for s in splits['splits'] if s['split']=='train'),None)
            if train:
                rows=fetch('https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=repo,config=train['config'],split='train',offset=0,length=3)))
                (folder/'viewer-preview.json').write_text(json.dumps(rows,indent=2)+'\n')
                result.update(features=rows.get('features'),preview_rows=len(rows.get('rows',[])),viewer_total_rows=rows.get('num_rows_total'),preview_revision_binding='Viewer_HEAD_only_not_admissible_as_pinned_training_rows')
        else:result['viewer_status']='unavailable_not_evidence_of_empty_dataset'
    (folder/'inspection.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/data/hf-candidates-v1.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for source in cfg['sources']:
        result=inspect(source,a.output);results.append(result)
        print(json.dumps(dict(repo=result['repo'],revision=result['pinned_revision_returned'],features=[f['name'] for f in result.get('features',[])],preview_rows=result.get('preview_rows',0),training_approved=False)),flush=True)
    (a.output/'inventory.json').write_text(json.dumps(dict(schema_version=1,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),sources=results,external_rows_admitted_to_training=0),indent=2)+'\n')


if __name__=='__main__':main()
