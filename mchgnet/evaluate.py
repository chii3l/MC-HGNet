"""Evaluate the validation-selected checkpoint; never select on the fixed test."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import Data,Bank
from .train import construct,configure,metrics
from .prepare import write_json
from .assets import sha256


def evaluate(data_folder,run,device='cpu'):
    run=Path(run)
    done=json.loads((run/'complete.json').read_text())
    if done.get('smoke'):raise ValueError('Smoke checkpoints cannot be evaluated as paper experiments')
    if sha256(run/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Checkpoint changed after selection')
    saved=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    config=saved['config'];configure(config['seed'])
    data=Data(data_folder,config['fold'],test=True)
    model=construct(data,config['encoder'],config['variant'],config,device)
    model.load_state_dict(saved['state'],strict=True)
    bank=Bank(data,config['encoder'],device)
    report={};predictions={}
    for split in ('train','validation','test'):
        ids=data.ids[split];p=bank.predict(model,ids)
        report[split]=metrics(data.y[ids],p,data.name)
        predictions[split+'_ids']=ids;predictions[split+'_prediction']=p
    write_json(run/'metrics.json',report)
    np.savez_compressed(run/'predictions.npz',**predictions)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(evaluate(a.data,a.run,a.device),indent=2))


if __name__=='__main__':main()
