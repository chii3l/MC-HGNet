"""Signal transformations and train-only condition PCA used by the paper."""
import numpy as np
from scipy.signal import cheby2, butter, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from .assets import asset


def signal_views(raw, crhe=False):
    source=np.asarray(raw)
    raw=np.asarray(raw,np.float64)
    stat_source=raw if crhe else source
    statistics=np.stack([stat_source.mean(-1),np.log(stat_source.std(-1).clip(1e-8))],-1).astype(np.float32)
    if crhe:
        detrended=raw-uniform_filter1d(raw,size=41,axis=-1,mode='reflect')
        z=sosfiltfilt(butter(5,[.5,8.],btype='bandpass',fs=40,output='sos'),
                     detrended,axis=-1,padtype='odd',padlen=33)
        span=np.ptp(z,axis=-1,keepdims=True)
        if not (span>1e-10).all(): raise ValueError('Degenerate CRHE waveform')
        canonical=((z-z.min(-1,keepdims=True))/span).astype(np.float32)
        views={40:canonical}
        for rate,up,down in ((50,5,4),(125,25,8)):
            x=resample_poly(canonical.astype(float),up,down,axis=-1,padtype='line')
            views[rate]=((x-x.min(-1,keepdims=True))/np.ptp(x,axis=-1,keepdims=True)).astype(np.float32)
    else:
        centered=raw-raw.mean(-1,keepdims=True)
        z=sosfiltfilt(cheby2(4,20,[.5,12],btype='bandpass',fs=50,output='sos'),centered,axis=-1)
        def normalize(x): return ((x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)).astype(np.float32)
        canonical=normalize(z)
        z125=sosfiltfilt(cheby2(4,20,[.5,12],btype='bandpass',fs=50,output='sos'),
                        raw-source.mean(-1,keepdims=True),axis=-1)
        views={50:canonical,125:normalize(resample_poly(z125,5,2,axis=-1))}
    return canonical,views,statistics


def condition_sentences(values, slots):
    out=[]
    for row in values:
        strings=[]
        for k,v in zip(slots,row):
            if not np.isfinite(v): text=f'Recorded {k}: unknown.'
            elif k=='age': text=f'Recorded age: {round(float(v),2):.2f} years.'
            elif k=='sex': text='Recorded sex: '+('female' if v==0 else 'male')+'.'
            elif k=='fitzpatrick': text=f'Recorded Fitzpatrick skin type: {int(v)}.'
            elif k=='warming': text='Recorded warming during measurement: '+('yes' if v==1 else 'no')+'.'
            elif k=='bmi': text=f'Recorded BMI: {round(float(v),2):.2f}.'
            else: raise ValueError('Supply the private-dataset field sentences explicitly: '+k)
            strings.append(text)
        out.append(strings)
    return out


def encode_text(strings, device='cpu'):
    """Frozen Qwen3-Embedding-0.6B, float32/eager, last token, L2 normalization."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    root=asset('qwen')
    tokenizer=AutoTokenizer.from_pretrained(root,padding_side='left',local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(root,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager').to(device).eval().requires_grad_(False)
    shape=np.asarray(strings).shape
    unique,inverse=np.unique(np.asarray(strings).flatten(),return_inverse=True)
    result=[]
    for start in range(0,len(unique),8):
        batch=tokenizer(unique[start:start+8].tolist(),padding=True,truncation=False,return_tensors='pt')
        if batch['input_ids'].shape[1]>1024: raise ValueError('Unexpected field text length')
        with torch.inference_mode():
            z=model(**{k:v.to(device) for k,v in batch.items()}).last_hidden_state[:,-1]
            result.append(torch.nn.functional.normalize(z.float(),p=2,dim=1).cpu().numpy())
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.concatenate(result)[inverse].reshape(*shape,1024).astype(np.float32)


def fit_fold(embeddings, available, statistics, train, active_slots=None):
    """No validation/test observations participate in fitting."""
    embeddings=np.asarray(embeddings)
    output=np.zeros((*available.shape,32),np.float32)
    audit=[]
    for k in range(available.shape[1]):
        ix=train[available[train,k]]
        if not len(ix):
            audit.append(None);continue
        x=embeddings[ix,k].astype(np.float64)
        center=x.mean(0)
        pca=PCA(n_components=min(32,len(ix),x.shape[1]),svd_solver='full').fit(x)
        z=(embeddings[:,k].astype(float)-center)@pca.components_.T
        rms=max(float(np.sqrt(np.mean(z[ix]**2))),1e-6)
        output[:,k,:z.shape[1]]=z/rms
        output[~available[:,k],k]=0
        audit.append(dict(center=center.tolist(),basis=pca.components_.tolist(),rms=rms))
    st=statistics.astype(float)
    center=np.median(st[train],0)
    scale=np.subtract(*np.percentile(st[train],[75,25],axis=0)).clip(1e-4)
    normal=np.clip((st-center)/scale,-8,8).astype(np.float32)
    fields=np.concatenate([output,available[...,None].astype(np.float32)],-1)
    if active_slots is not None:
        fields[:,[k for k in range(available.shape[1]) if k not in active_slots]]=0
    return fields,normal,dict(fit_units=train.tolist(),condition=audit,
        statistics_center=center.tolist(),statistics_scale=scale.tolist())


def histories(frame, folds, crhe=False):
    """Latest completed same-person/day/split record; no historical targets."""
    n=len(frame); slots=5
    result=[]
    for fold in folds:
        index=np.repeat(np.arange(n)[:,None],slots,1)
        mask=np.zeros((n,slots),bool);gap=np.zeros((n,slots),np.float32)
        for split,ids in fold.items():
            groups={}
            for u in ids: groups.setdefault((str(frame.subject.iloc[u]),str(frame.day.iloc[u])),[]).append(u)
            for group in groups.values():
                seen=[]
                for u in sorted(group,key=lambda u:(frame.start.iloc[u],u)):
                    eligible=[h for h in seen if frame.end.iloc[h]<frame.start.iloc[u] and
                        (not crhe or frame.timestamp_seconds.iloc[u]-frame.timestamp_seconds.iloc[h]>=300)]
                    prior=eligible[-1:] if crhe else eligible[-slots:]
                    if prior:
                        index[u,:len(prior)]=prior
                        mask[u,len(prior)-1]=True
                        for j,h in enumerate(prior):
                            gap[u,j]=((frame.timestamp_seconds.iloc[u]-frame.timestamp_seconds.iloc[h]) if crhe
                                      else (frame.start.iloc[u]-frame.end.iloc[h]))/86400
                    seen.append(u)
        gap=np.where(mask,gap,0).astype(np.float32)
        result.append(dict(index=index,mask=mask,gap=gap))
    return result
