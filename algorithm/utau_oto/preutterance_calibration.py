"""Human-supervised correction of method 1's preutterance only.

The regressor sees method 1's own five lines and numerical WAV features. It
never consumes Vision or Alignment predictions. Missing models leave method 1
unchanged, and all corrections remain within the other four lines.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np

from .experimental.config import Config
from .experimental.features import extract_features
from .experimental.phonemes import parse_alias


def vector(entry,phones,features,count):
    times=np.asarray(features['times'],float)
    absolute=float(entry['offset']+entry['preutterance'])
    k=int(np.argmin(abs(times-absolute)))
    values=dict(initial=int(entry['mora_index']==0),index=entry['mora_index'],count=count,
        consonant=phones['consonant'] or 'vowel',previous=phones['previous'],vowel=phones['vowel'],
        offset=entry['offset'],pre=entry['preutterance'],fixed=entry['consonant'],
        overlap=entry['overlap'],cutoff=entry['cutoff'],absolute=absolute)
    for name in ('power_db','flux','periodicity'):
        series=np.asarray(features[name],float)
        for delta in (-100,-50,-25,0,25,50,100):
            j=int(np.clip(k+round(delta/5),0,len(series)-1))
            values[f'{name}_{delta}']=float(series[j])
        values[name+'_local_max']=float(np.max(series[max(0,k-15):min(len(series),k+16)]))
    alignment=np.asarray(features['alignment'],float)
    for i,value in enumerate(np.mean(alignment,axis=0)[:20]):values[f'mean_mfcc_{i}']=float(value)
    for i,value in enumerate(np.mean(alignment[max(0,k-5):min(len(alignment),k+6)],axis=0)[:20]):
        values[f'local_mfcc_{i}']=float(value)
    mel=np.asarray(features['mel'],float)
    for delta in range(-120,121,10):
        j=int(np.clip(k+round(delta/5),0,mel.shape[1]-1))
        for band in range(10):
            chunk=mel[band*8:(band+1)*8,j]
            values[f'mel_{band}_{delta}']=float(chunk.mean())
    for name in ('power_db','flux','periodicity'):
        series=np.asarray(features[name],float)
        for delta in (-75,-50,-25,0,25,50,75):
            j=int(np.clip(k+round(delta/5),2,len(series)-3))
            values[f'{name}_derivative_{delta}']=float(series[j+2]-series[j-2])
        lo=max(0,k-20);hi=min(len(series),k+21)
        local=series[lo:hi]
        values[name+'_local_std']=float(local.std())
        values[name+'_strongest_relative_ms']=float((np.argmax(local)+lo-k)*5)
    return values


@lru_cache(maxsize=2)
def load_model(path):
    import joblib
    model=joblib.load(path)
    if model.get('format')!='uto-existing-calibration-1':raise ValueError('Unsupported calibration model')
    return model


def calibrate_entries(entries,wav_path,model_path=None,cache_dir=None):
    """Return corrected entries and a status; WAV and four other lines are untouched."""
    app=Path(__file__).resolve().parents[1]
    model_path=Path(model_path) if model_path else app/'models/existing_calibration.joblib'
    if not model_path.is_file():return entries,dict(status='unavailable',reason='Calibration model missing')
    try:
        model=load_model(str(model_path))
        features=extract_features(wav_path,Config().features,
                                  cache_dir if cache_dir is not None else Path(wav_path).parent/'キャッシュ'/'音響特徴'/'existing')
        prepared=[];locations=[]
        count=max((int(e['mora_index']) for e in entries),default=-1)+1
        for i,entry in enumerate(entries):
            try:phones=parse_alias(entry['alias'])
            except ValueError:continue
            prepared.append(vector(entry,phones,features,count));locations.append(i)
        if not prepared:return entries,dict(status='unavailable',reason='No VCV alias to calibrate')
        correction=model['regressor'].predict(model['vectorizer'].transform(prepared))
        revised=[dict(e) for e in entries]
        for i,delta in zip(locations,correction):
            entry=revised[i]
            original=float(entry['preutterance'])
            adjusted=float(np.clip(original+float(delta),entry['overlap'],entry['consonant']))
            entry['preutterance']=round(adjusted,3)
            entry['detected_absolute_ms']=dict(entry.get('detected_absolute_ms',{}),
                vowel_transition=entry['offset']+entry['preutterance'])
            lines=dict(entry.get('line_diagnostics',{}))
            if 'preutterance' in lines:
                diagnostic=dict(lines['preutterance'])
                diagnostic['selected_absolute_ms']=round(entry['offset']+entry['preutterance'],3)
                diagnostic['reason']=diagnostic.get('reason','')+' / 2音源の手動設定による校正'
                diagnostic['calibration_delta_ms']=round(adjusted-original,3)
                lines['preutterance']=diagnostic
                entry['line_diagnostics']=lines
            entry['reasons']=list(entry.get('reasons',[]))+['先行発声: 2音源の手動設定による校正']
        return revised,dict(status='ok',calibrated=len(locations))
    except Exception as exc:
        return entries,dict(status='unavailable',reason=str(exc))
