"""Write the two focus-split corpus configs: uniform extra variants (control) and weak-pair weighted ones."""
import json
from pathlib import Path
import shallow

SEED=314159
UNIFORM=8
WEIGHTED=16

def config(name,weighted):
    families=dict(sorted(shallow.RF_SPLITS.items()))
    extra={}
    for family,split in families.items():
        if split!='train':continue
        if family.startswith('rf_two_'):
            outer,inner=family.split('_')[2:4]
            extra[family]=WEIGHTED if weighted and shallow.rf_weak(outer,inner) else UNIFORM
        else:
            extra[family]=UNIFORM
    return dict(schema_version=1,name=name,seed=SEED,variants_per_family=14,optimizations=['O1'],representation='pcode-plus-ssa-v1',families=families,extra_variants=extra)

def main():
    for name,weighted in (('forge-shallow-focus-v1-uniform',False),('forge-shallow-focus-v1-weighted',True)):
        path=Path('configs/corpus')/(name.replace('forge-shallow-','shallow-')+'.json')
        path.write_text(json.dumps(config(name,weighted),indent=2)+'\n')
        print(path)

if __name__=='__main__':main()
