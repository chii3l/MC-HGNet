"""Portable measurement-level data and causal shared-encoder batching."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch


class Data:
    def __init__(self,folder,fold=0,test=False):
        folder=Path(folder); self.folder=folder;self.fold=fold
        self.spec=json.loads((folder/'spec.json').read_text())
        self.name=self.spec['dataset']; self.test=test
        self.ids={k:np.asarray(v,np.int64) for k,v in json.loads((folder/f'fold{fold}/fold.json').read_text()).items()}
        self.frame=pd.read_csv(folder/'units.csv',dtype={'subject':str})
        self.subjects=self.frame.subject.to_numpy(str)
        self.y=np.load(folder/'labels.npy',allow_pickle=False).astype(float)
        if self.y.ndim==1:self.y=self.y[:,None]
        norm=json.loads((folder/f'fold{fold}/normalizer.json').read_text())
        if norm['fit_units']!=self.ids['train'].tolist():raise ValueError('Normalization fit indices differ')
        self.center=np.asarray(norm['target_center']);self.scale=np.asarray(norm['target_scale'])
        with np.load(folder/f'fold{fold}/inputs.npz',allow_pickle=False) as z:
            self.fields=z['fields'].copy(); self.statistics=z['statistics'].copy()
            self.history={k:z[k].copy() for k in ('index','mask','gap')}
        allowed=np.concatenate([self.ids[k] for k in (('train','validation','test') if test else ('train','validation'))])
        self.allowed=set(allowed.tolist())
        withheld=np.setdiff1d(np.arange(len(self.y)),allowed)
        self.y[withheld]=np.nan
        self.fields[withheld]=0;self.statistics[withheld]=0
        self.history['mask'][withheld]=False
        self.id_map={s:i for i,s in enumerate(sorted(set(self.subjects[self.ids['train']]))) }
        self.person=np.array([self.id_map.get(s,-1) for s in self.subjects],np.int64)
        self.structure=None
        if (folder/'structure.npz').exists():
            with np.load(folder/'structure.npz',allow_pickle=False) as z:
                self.structure={k:z[k].copy() for k in z.files}
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            if set(self.ids[a]) & set(self.ids[b]):raise ValueError('Overlapping splits')
        assignment={u:k for k,ids in self.ids.items() for u in ids}
        for u in allowed:
            for h in self.history['index'][u,self.history['mask'][u]]:
                if h not in self.allowed or assignment[h]!=assignment[u] or self.subjects[h]!=self.subjects[u]:
                    raise ValueError('Invalid historical reference')
                if self.frame.day.iloc[u]!=self.frame.day.iloc[h] or self.frame.end.iloc[h]>=self.frame.start.iloc[u]:
                    raise ValueError('Noncausal/overlapping historical reference')


def median_pool(values,mask):
    ordered=values.masked_fill(~mask[:,:,None],float('inf')).sort(1).values
    count=mask.sum(1)
    lo=((count-1)//2)[:,None,None].expand(-1,1,values.shape[-1])
    hi=(count//2)[:,None,None].expand(-1,1,values.shape[-1])
    return .5*(ordered.gather(1,lo)+ordered.gather(1,hi)).squeeze(1)


class Bank:
    def __init__(self,data,encoder,device='cpu'):
        self.data,self.encoder,self.device=data,encoder,device
        rate=125 if encoder in ('papagei_pretrained','anyppg') else 50
        filename='wave.npy' if encoder in ('moderntcn','timefilter','srformer') else f'wave{rate}.npy'
        self.wave=torch.as_tensor(np.load(data.folder/filename,allow_pickle=False),device=device)
        self.statistics=torch.as_tensor(data.statistics,device=device)
        self.fields=torch.as_tensor(data.fields,device=device)
        self.person=torch.as_tensor(data.person,device=device)
        self.target=torch.as_tensor((data.y-data.center)/data.scale,dtype=torch.float32,device=device)
        self.structure=data.structure
        if self.structure is not None:
            self.structure={k:torch.as_tensor(v,device=device) for k,v in self.structure.items()}

    def encode_units(self,model,units):
        units=np.asarray(units,np.int64)
        u=torch.as_tensor(units,device=self.device)
        if self.structure is None:
            if model.name=='srformer':
                # Preserve native sorted-record chunks, including their training BN batches.
                unique,inverse=np.unique(units,return_inverse=True)
                index=torch.as_tensor(unique,device=self.device)
                loc=torch.as_tensor(inverse,device=self.device)
                tokens,native=model.encode(self.wave[index],self.person[index])
                return tokens[loc],native[loc]
            tokens,native=model.encode(self.wave[u])
            return tokens,native
        index=self.structure['index'][u];mask=self.structure['record_mask'][u]
        records=index[mask]
        if model.name=='srformer':
            unique,inverse=records.unique(sorted=True,return_inverse=True)
            owner=self.structure['owner'][unique]
            tok,nat=model.encode(self.wave[unique],self.person[owner])
            tokens=torch.cat([nat,tok.flatten(1)],-1)[inverse]
        else:
            tok,_=model.encode(self.wave[records]);tokens=tok.flatten(1)
        packed=tokens.new_zeros(*index.shape,tokens.shape[-1])
        packed[mask]=tokens
        if self.data.spec['pooling']=='capture_median':
            membership=self.structure['membership'][u]
            captures=torch.bmm(membership,packed)
            if model.name=='srformer':captures=captures/membership.sum(-1,keepdim=True).clamp_min(1.)
            pooled=median_pool(captures,self.structure['capture_mask'][u])
        else:
            pooled=packed.sum(1)/mask.sum(1)[:,None]
        if model.name=='srformer':
            width=model.length//2
            return pooled[:,width:].reshape(len(units),model.c,64),pooled[:,:width]
        return pooled.reshape(len(units),model.c,64),None

    def select(self,units,history):
        units=np.asarray(units,np.int64)
        if not set(units)<=self.data.allowed:raise ValueError('Withheld observations requested')
        if not history:
            return units,units[:,None],np.arange(len(units))[:,None],None,None
        h=self.data.history;mask=h['mask'][units]
        safe=np.where(mask,h['index'][units],units[:,None])
        ix=np.concatenate([units[:,None],safe],1)
        union=np.array(list(dict.fromkeys(ix.flatten().tolist())),np.int64)
        lookup={u:i for i,u in enumerate(union)}
        inverse=np.array([[lookup[u] for u in row] for row in ix],np.int64)
        return union,ix,inverse,mask,h['gap'][units]

    def forward(self,model,units):
        union,ix,inverse,mask,gap=self.select(units,model.variant not in ('baseline','no_history'))
        if model.name=='timefilter' and model.training and mask is not None:
            current,_=self.encode_units(model,units)
            aux=model.encoder.auxiliary_loss
            missing=np.array([u for u in union if u not in set(units)],np.int64)
            ordered=np.r_[units,missing];lookup={u:i for i,u in enumerate(ordered)}
            values=torch.cat([current,self.encode_units(model,missing)[0]]) if len(missing) else current
            model.encoder.auxiliary_loss=aux
            tokens=values[torch.as_tensor([lookup[u] for u in union],device=self.device)]
            native=None
        else:
            tokens,native=self.encode_units(model,union)
        loc=torch.as_tensor(inverse,device=self.device)
        selected=torch.as_tensor(ix,device=self.device)
        u=torch.as_tensor(units,device=self.device)
        pending=None if mask is None else (tokens[loc],self.statistics[selected],self.fields[selected].flatten(2),
            torch.as_tensor(mask,device=self.device),torch.as_tensor(gap,device=self.device))
        output=model.finish(tokens[loc[:,0]],self.statistics[u],self.fields[u],pending)
        if native is not None:
            model.identity=torch.cat([native[loc[:,0]],self.statistics[u].flatten(1),self.fields[u].flatten(1)],-1)
        return output

    @torch.no_grad()
    def predict(self,model,units):
        model.eval();predictions=[]
        for start in range(0,len(units),128 if model.name=='srformer' else 64):
            size=128 if model.name=='srformer' else 64
            predictions.append(self.forward(model,units[start:start+size]).cpu().numpy())
        return np.concatenate(predictions)*self.data.scale+self.data.center
