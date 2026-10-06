"""Independent human-supervised residual correction for method 3.

The input contains only method 3's own estimate and numerical WAV features.
Neither method 1 nor method 2 is an input or a training target.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
from .result import PreutteranceEstimate


def vector(index,count,phones,estimate,confidence,metadata,features):
    times=np.asarray(features['times'],float)
    estimate=float(estimate)
    k=int(np.argmin(abs(times-estimate)))
    values=dict(index=index,count=count,initial=int(index==0),
        consonant=phones['consonant'] or 'vowel',previous=phones['previous'],vowel=phones['vowel'],
        estimated_ms=estimate,confidence=float(confidence),duration_ms=float(features['duration_ms']))
    for key in ('best_path_cost','second_best_path_cost','path_margin','template_fit','duration_fit',
                'boundary_contrast','acoustic_phase_shift_ms'):
        if isinstance(metadata.get(key),(float,int)):values[key]=float(metadata[key])
    rhythm=metadata.get('recording_rhythm')
    if isinstance(rhythm,dict):
        for key in ('period_ms','phase_ms','score','tail_beats'):
            if isinstance(rhythm.get(key),(float,int)):values['rhythm_'+key]=float(rhythm[key])
        if all(isinstance(rhythm.get(key),(float,int)) for key in ('onset_ms','phase_ms','period_ms')):
            center=rhythm['onset_ms']+rhythm['phase_ms']+index*rhythm['period_ms']
            values['rhythm_boundary_delta_ms']=estimate-float(center)
    for name in ('power_db','flux','periodicity'):
        series=np.asarray(features[name],float)
        for delta in (-100,-75,-50,-25,0,25,50,75,100):
            j=int(np.clip(k+round(delta/5),0,len(series)-1))
            values[f'{name}_{delta}']=float(series[j])
        values[name+'_local_max']=float(np.max(series[max(0,k-20):min(len(series),k+21)]))
    alignment=np.asarray(features['alignment'],float)
    for i,value in enumerate(np.mean(alignment,axis=0)[:20]):values[f'mean_mfcc_{i}']=float(value)
    for i,value in enumerate(np.mean(alignment[max(0,k-5):min(len(alignment),k+6)],axis=0)[:20]):
        values[f'local_mfcc_{i}']=float(value)
    return values


@lru_cache(maxsize=4)
def load_calibration(path):
    import joblib
    artifact=joblib.load(path)
    if artifact.get('format')!='uto-alignment-calibration-1':
        raise ValueError('Unsupported alignment calibration')
    return artifact


def calibrate(estimate,features,index,count,phones,template_path):
    if estimate.status!='ok' or estimate.position_ms is None:return estimate
    template_path=Path(template_path)
    path=template_path.with_name('alignment_calibration.joblib')
    if not path.is_file():return estimate
    try:
        model=load_calibration(str(path))
        digest=hashlib.sha256(template_path.read_bytes()).hexdigest()
        if digest!=model['template_sha256']:return estimate
        values=vector(index,count,phones,estimate.position_ms,estimate.confidence,estimate.metadata,features)
        delta=float(model['regressor'].predict(model['vectorizer'].transform([values]))[0])
        raw=float(estimate.position_ms)
        corrected=float(np.clip(raw+delta,0,float(features['duration_ms'])))
        if not np.isfinite(corrected):return estimate
        metadata=dict(estimate.metadata,raw_position_ms=raw,calibration_delta_ms=corrected-raw,
                      calibration='human supervised method 3 residual')
        candidates=[dict(position_ms=corrected,source='calibrated')]+list(estimate.candidates)
        return PreutteranceEstimate('alignment',corrected,estimate.confidence,candidates,metadata)
    except Exception as exc:
        estimate.metadata=dict(estimate.metadata,calibration_unavailable=str(exc))
        return estimate
