"""Measurement-weighted regression metrics. Glucose is reported in mg/dL."""
import numpy as np


def zone_a(truth,prediction):
    # Parkes Type 1 A/B boundary, Pfutzner et al. 2013, Table 1.
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3876371/
    y,p=np.asarray(truth,float),np.asarray(prediction,float)
    if ((y<0)|(p<0)|(y>550)|(p>550)).any():
        raise ValueError('Outside published Parkes grid; predictions are not clipped')
    upper=np.array([(0,50),(30,50),(140,170),(280,380),(430,550)],float)
    lower=np.array([(50,0),(50,30),(170,145),(385,300),(550,450)],float)
    above=p>np.interp(y,upper[:,0],upper[:,1])+1e-9
    below=(y>50+1e-9)&(p<np.interp(y,lower[:,0],lower[:,1])-1e-9)
    return float(100*(~(above|below)).mean())


def regression(truth,prediction,dataset):
    y,p=np.asarray(truth,float),np.asarray(prediction,float)
    if y.ndim==1:y=y[:,None]
    if p.ndim==1:p=p[:,None]
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metrics require all measurement predictions, without dropping rows')
    if dataset=='glucose':y,p=y*18.,p*18.
    names=['SBP','DBP'] if dataset=='bp_joint' else [dict(glucose='glucose',openox_spo2='SpO2',openox_sao2='SaO2')[dataset]]
    result={}
    for j,name in enumerate(names):
        e=p[:,j]-y[:,j]
        result[name]=dict(mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),n=len(y))
        if dataset=='glucose':result[name]['zone_a_pct']=zone_a(y[:,j],p[:,j])
    return result
