"""Prepare the two public datasets from official original downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from .assets import sha256
from .preprocessing import signal_views, condition_sentences, encode_text, fit_fold, histories

PROTOCOL=Path(__file__).resolve().parent/'resources/protocol'


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_crhe(root):
    from .crhe import load_raw,clean,make_partition
    hashes=json.loads((PROTOCOL/'crhe_source_hashes.json').read_text())
    for name,digest in hashes.items():
        if sha256(root/name)!=digest: raise ValueError('Official CRHE file differs: '+name)
    grouped,_=load_raw(root)
    kept,excluded,duplicates=clean(grouped)
    frame=pd.DataFrame([{k:r[k] for k in ('unit','subject','timestamp','date','age','sex_code')} for r in kept])
    partition=make_partition(frame[['unit','subject','timestamp','date']])
    raw=np.stack([r['waveform'] for r in kept])
    y=np.array([[r['sbp'],r['dbp']] for r in kept],np.float32)
    frame['timestamp_seconds']=pd.to_datetime(frame.timestamp).astype('datetime64[ns]').astype('int64')/1e9
    frame['day']=frame.date
    frame['start']=frame.timestamp_seconds-68
    frame['end']=frame.timestamp_seconds+68
    values=frame[['age','sex_code']].to_numpy(float)
    folds=[f['units'] for f in partition['folds']]
    spec=dict(dataset='bp_joint',task='regression',channels=['green','red','ir'],
        wavelengths=[550,660,880],slots=['age','sex'],late_text_dim=66,outputs=2,
        pooling='mean',input_samples=320,sampling_hz=40,seconds=8,targets=['SBP','DBP'],units='mmHg')
    if (len(frame),len(partition['development_units']),len(folds[0]['test']))!=(4178,3388,788):
        raise ValueError('CRHE cohort mismatch')
    return spec,frame,raw,values,{'bp_joint':y},folds


def load_openox(root):
    # WFDB 4.2 expects object-backed strings on pandas 3 as on pandas 2.
    if hasattr(pd.options.future, 'infer_string'): pd.options.future.infer_string=False
    import wfdb
    protocol=json.loads((PROTOCOL/'openox_windows.json').read_text())
    windows={r['key']:r for r in protocol['windows']}
    n=protocol['array_length']
    patient=pd.read_csv(root/'patient.csv').set_index('patient_id')
    encounter=pd.read_csv(root/'encounter.csv').set_index('encounter_id')
    blood=pd.read_csv(root/'bloodgas.csv',low_memory=False)
    pulse=pd.read_csv(root/'pulseoximeter.csv',low_memory=False)
    for k in ('sample','so2'): blood[k]=pd.to_numeric(blood[k],errors='coerce')
    for k in ('sample_number','saturation','device'): pulse[k]=pd.to_numeric(pulse[k],errors='coerce')
    blood=blood[blood['sample'].notna() & (blood['sample']%1==0)]
    pulse=pulse[pulse['sample_number'].notna() & (pulse['sample_number']%1==0)]
    raw=np.zeros((n,2,300),np.float32); values=np.full((n,4),np.nan)
    ys={k:np.full((n,1),np.nan,np.float32) for k in ('openox_spo2','openox_sao2')}
    rows=[dict(unit=i,subject='excluded',day='excluded',start=0.,end=0.) for i in range(n)]
    seen=set()
    for (session,sample), assays in blood.groupby(['encounter_id','sample'],sort=True):
        key=hashlib.sha256((str(session)+'|'+str(int(sample))).encode()).hexdigest()
        if key not in windows: continue
        window=windows[key]; u=window['unit'];seen.add(key)
        enc=encounter.loc[session];person=patient.loc[enc.patient_id]
        path=root/'waveforms'/str(session)[0]/(str(session)+'_ppg')
        header=wfdb.rdheader(str(path))
        names=[s.lower() for s in header.sig_name]
        channels=[next(i for i,s in enumerate(names) if s in aliases)
                  for aliases in (('red','red signal'),('ir','ir signal'))]
        if header.fs!=86: raise ValueError('Expected OpenOx 1.1.1 86 Hz PPG')
        start=window['start_sample']
        signal=wfdb.rdrecord(str(path),sampfrom=start,sampto=start+516,channels=channels,physical=True).p_signal.T
        if not np.isfinite(signal).all() or (signal.std(-1)<1e-10).any(): raise ValueError('Invalid retained PPG')
        raw[u]=resample_poly(signal,50,86,axis=-1,padtype='line').astype(np.float32)
        sex=str(person.assigned_sex).strip().lower()
        values[u]=[enc.age_at_encounter,{'female':0,'f':0,'male':1,'m':1}.get(sex,np.nan),enc.fitzpatrick,enc.warming]
        valid=assays.loc[np.isfinite(assays.so2),'so2']
        if valid.empty or valid.max()-valid.min()>1.000001: raise ValueError('Arterial cohort mismatch')
        ys['openox_sao2'][u,0]=valid.median()
        devices=pulse[(pulse.encounter_id==session)&(pulse.sample_number==sample)&
                      pulse.saturation.between(50,100)&pulse.device.notna()]
        if devices.empty or devices.device.duplicated().any(): raise ValueError('Device target mismatch')
        ys['openox_spo2'][u,0]=devices.saturation.mean()
        day=str(enc.encounter_date)
        origin=pd.Timestamp(day).to_pydatetime().toordinal()*86400.
        rows[u]=dict(unit=u,subject=str(enc.patient_id),day=day,
            start=origin+window['reference_start_clock_seconds'],
            end=origin+window['reference_end_clock_seconds'])
    if seen!=set(windows): raise ValueError('Missing retained OpenOx encounters/samples')
    folds=[json.loads((PROTOCOL/f'openox_fold{i}.json').read_text()) for i in range(5)]
    spec=dict(dataset='openox',task='regression',channels=['red','infrared'],wavelengths=[None,None],
        slots=['age','sex','fitzpatrick','warming'],active_slots=[0,1],late_text_dim=132,
        outputs=1,pooling='mean',input_samples=300,sampling_hz=50,seconds=6,units='percentage_points')
    return spec,pd.DataFrame(rows),raw,values,ys,folds


def prepare(dataset,root,output,device='cpu',embeddings_path=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory to avoid mixing preprocessing runs')
    spec,frame,raw,values,targets,folds=(load_crhe if dataset=='crhe' else load_openox)(Path(root))
    canonical,views,statistics=signal_views(raw,crhe=dataset=='crhe')
    strings=condition_sentences(values,spec['slots'])
    available=np.isfinite(values)
    embeddings=np.load(embeddings_path,allow_pickle=False) if embeddings_path else encode_text(strings,device)
    if embeddings.shape!=(*values.shape,1024): raise ValueError('Field embedding shape mismatch')
    history=histories(frame,folds,crhe=dataset=='crhe')
    fitted=[]
    for ids in folds:
        train=np.array(ids['train'],dtype=np.int64)
        fitted.append(fit_fold(embeddings,available,statistics,train,spec.get('active_slots')))
    for name,y in targets.items():
        folder=output/name;folder.mkdir(parents=True)
        current_spec=dict(spec,dataset=name)
        write_json(folder/'spec.json',current_spec)
        frame.to_csv(folder/'units.csv',index=False)
        np.save(folder/'wave.npy',canonical,allow_pickle=False)
        for hz,view in views.items(): np.save(folder/f'wave{hz}.npy',view,allow_pickle=False)
        np.save(folder/'labels.npy',y,allow_pickle=False)
        np.save(folder/'field_embeddings.npy',embeddings,allow_pickle=False)
        write_json(folder/'condition_sentences.json',strings)
        for f,ids in enumerate(folds):
            train=np.array(ids['train'],dtype=np.int64)
            fields,stats,parameters=fitted[f]
            norm=dict(parameters)
            center=y[train].astype(float).mean(0);scale=y[train].astype(float).std(0).clip(1e-6)
            if name.startswith('openox_'):
                center=y[train].mean(0);scale=y[train].std(0)
            fold=folder/f'fold{f}';fold.mkdir()
            write_json(fold/'fold.json',ids)
            norm.update(target_center=center.tolist(),target_scale=scale.tolist())
            write_json(fold/'normalizer.json',norm)
            np.savez_compressed(fold/'inputs.npz',fields=fields,statistics=stats,**history[f])
        write_json(folder/'preparation.json',dict(dataset=name,observations=int(sum(len(v) for v in folds[0].values())),
            test_count=len(folds[0]['test']),preprocessing='fit_training_fold',fields=spec['slots'],
            active_slots=spec.get('active_slots'),embeddings_sha256=sha256(folder/'field_embeddings.npy')))
    return list(targets)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['crhe','openox'],required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--field-embeddings',type=Path,help='Optional locally generated frozen-Qwen field embeddings')
    args=p.parse_args()
    print(prepare(args.dataset,args.raw,args.out,args.device,args.field_embeddings))


if __name__=='__main__': main()
