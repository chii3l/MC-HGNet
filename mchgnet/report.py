"""Aggregate five fixed-test evaluations as mean +/- sample SD, without ensembling."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();groups={}
    for root in a.runs:
        for file in root.rglob('metrics.json'):
            config=json.loads((file.parent/'config.json').read_text())
            if config.get('smoke'):continue
            values=json.loads(file.read_text())
            for split,targets in values.items():
                for target,metrics in targets.items():
                    for metric,value in metrics.items():
                        if metric=='n':continue
                        key=(config['dataset'],config['encoder'],config['variant'],target,split,metric)
                        cell=groups.setdefault(key,{})
                        if config['fold'] in cell:raise ValueError('Duplicate fold: '+str(key))
                        cell[config['fold']]=value
    rows=[]
    for key,folds in sorted(groups.items()):
        if set(folds)!=set(range(5)):raise ValueError('Five completed folds required: '+str(key))
        x=list(folds.values())
        rows.append(dict(zip(('dataset','encoder','variant','target','split','metric'),key),
                         mean=float(np.mean(x)),sample_sd=float(np.std(x,ddof=1)),folds=5))
    if not rows:raise ValueError('No completed evaluations')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(a.out)


if __name__=='__main__':main()
