"""Complete-epoch training with validation-only checkpoint selection."""
import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
from .data import Data,Bank
from .model import MCHGNet
from .prepare import write_json
from .assets import sha256


def configure(seed=2026):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False


def recipe(dataset,encoder,variant,fold):
    p=Path(__file__).resolve().parent/'resources/configs/paper_recipes.json'
    key=f'{dataset}/{encoder}/{"baseline" if variant=="baseline" else "full"}/fold{fold}'
    return json.loads(p.read_text())[key]


def construct(data,encoder,variant,config,device):
    if encoder=='srformer':
        if any((data.person[ids]<0).any() for split,ids in data.ids.items()
               if split!='test' or data.test):
            raise ValueError('SRFormer requires evaluation subjects represented in its training ID map')
        data.y=data.y.astype(np.float32)
        data.center=data.center.astype(np.float32)
        data.scale=data.scale.astype(np.float32)
    spectra=None
    if encoder=='srformer' and data.name=='glucose':
        with np.load(data.folder/'spectra.npz',allow_pickle=False) as z:
            spectra=z['led'],z['pd']
    model=MCHGNet(data.spec,encoder,variant,config['seed'],len(data.id_map),config['srformer_dropout'],spectra).to(device)
    torch.backends.cudnn.allow_tf32=not(encoder=='sigmappg' and data.name!='bp_joint')
    return model


def schedule(ids,fold,config,srformer=False):
    rng=np.random.RandomState(config['seed']+(0 if srformer else fold*100003))
    s=config['settings']
    for epoch in range(s['epochs']):
        order=rng.permutation(np.asarray(ids,np.int64))
        for i in range(0,len(order),s['batch']):yield order[i:i+s['batch']]


def metrics(y,p,dataset):
    from .metrics import regression
    return regression(y,p,dataset)


def fit(folder,encoder,variant,fold,out,device='cpu',max_updates=None):
    data=Data(folder,fold,test=False)
    config=recipe(data.name,encoder,variant,fold)
    config['encoder']=encoder;config['variant']=variant;config['fold']=fold;config['dataset']=data.name
    config['smoke']=max_updates is not None
    s,p=config['settings'],config['profile']
    if math.ceil(len(data.ids['train'])/s['batch'])!=s['steps']:raise ValueError('Training cohort differs from the paper')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Completed run exists; choose another output')
    configure(config['seed'])
    model=construct(data,encoder,variant,config,device)
    bank=Bank(data,encoder,device)
    sr=encoder=='srformer'
    if sr:
        opt=torch.optim.Adam(model.parameters(),lr=s['lr'],weight_decay=s['weight_decay'])
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=s['step_lr_epochs'],gamma=s['step_lr_gamma'])
    else:
        main=[v for n,v in model.named_parameters() if not(model.is_pretrained and n.startswith('encoder.'))]
        groups=[dict(params=main,lr=p['lr'],base_lr=p['lr'])]
        if model.is_pretrained:groups.append(dict(params=list(model.encoder.parameters()),lr=p['encoder_lr'],base_lr=p['encoder_lr']))
        opt=torch.optim.AdamW(groups,weight_decay=p['weight_decay']);scheduler=None
    write_json(out/'config.json',config)
    best=float('inf');best_step=0;logs=[];first=1
    if (out/'resume.pt').exists():
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=True)
        if saved['config']!=config:raise ValueError('Resume config mismatch')
        model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer'])
        if scheduler:scheduler.load_state_dict(saved['scheduler'])
        best,best_step,logs,first=saved['best'],saved['best_step'],saved['logs'],saved['step']+1
        torch.set_rng_state(saved['cpu_rng']);random.setstate(saved['python_rng'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
    for step,units in enumerate(schedule(data.ids['train'],fold,config,sr),1):
        if step<first:continue
        active=sr or not model.is_pretrained or step>s['warmup_updates']
        model.set_active(active);model.train();opt.zero_grad(set_to_none=True)
        if not sr:
            interval=(step-1)//s['lr_interval'];intervals=math.ceil(s['updates']/s['lr_interval'])
            factor=.1+.9*.5*(1+math.cos(math.pi*interval/max(intervals-1,1)))
            for group in opt.param_groups:group['lr']=group['base_lr']*factor
        total=0.
        micro=len(units) if sr else s['microbatch']
        for start in range(0,len(units),micro):
            subset=units[start:start+micro]
            prediction=bank.forward(model,subset)
            target=bank.target[subset]
            if sr:
                from .srformer import identity_contrast
                w=s['identity_contrast_weight']
                loss=(1-w)*(prediction-target).square().mean()+w*identity_contrast(model.identity,bank.person[subset])
            else:
                def objective(q):
                    x=(1-p['mse'])*F.smooth_l1_loss(q,target,beta=p['beta'],reduction='sum')+p['mse']*F.mse_loss(q,target,reduction='sum')
                    return x/(len(units)*data.y.shape[1])
                loss=objective(prediction)+p['aux']*sum(objective(q) for q in model.parts)+model.auxiliary()*len(subset)/len(units)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),s['gradient_clip'] if sr else s['clip'],
                                      error_if_nonfinite=True,foreach=False if sr else None)
        opt.step()
        if scheduler and step%s['steps']==0:scheduler.step()
        stopping=max_updates is not None and step>=max_updates
        if step%s['evaluate_every']==0 or step==s['updates'] or stopping:
            cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            py=random.getstate()
            val=bank.predict(model,data.ids['validation'])
            score=float(np.abs(val-data.y[data.ids['validation']]).mean())
            torch.set_rng_state(cpu);random.setstate(py)
            if cuda:torch.cuda.set_rng_state_all(cuda)
            eligible=sr or step>s['warmup_updates'] or stopping
            if eligible and score<best-1e-8:
                best,best_step=score,step
                torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),out/'best.pt')
            row=dict(step=step,epoch=step/s['steps'],loss=total,validation_mae=score,
                     best_validation_mae=best if math.isfinite(best) else None,
                     learning_rate=opt.param_groups[0]['lr'])
            logs.append(row);write_json(out/'history.json',logs)
            print(json.dumps(row),flush=True)
            torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                config=config,step=step,best=best,best_step=best_step,logs=logs,cpu_rng=cpu,cuda_rng=cuda,python_rng=py),out/'resume.pt')
        if stopping:break
    write_json(out/'complete.json',dict(completed=True,smoke=max_updates is not None,updates=step,
        selected_step=best_step,validation_mae=best,checkpoint_sha256=sha256(out/'best.pt')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--encoder',choices=['moderntcn','srformer','pulseppg','papagei_pretrained','anyppg','sigmappg','timefilter'],required=True)
    p.add_argument('--variant',choices=['baseline','full','no_history','no_pairs','no_graph'],default='full')
    p.add_argument('--fold',type=int,choices=range(5),default=0)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-updates',type=int,help='Smoke run only, never a paper result')
    a=p.parse_args();fit(a.data,a.encoder,a.variant,a.fold,a.out,a.device,a.max_updates)


if __name__=='__main__':main()
