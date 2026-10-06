"""Independent, versioned features shared only by estimators 2 and 3."""
from dataclasses import asdict
from pathlib import Path
import hashlib, io, json
import librosa
import numpy as np
from ..audio import read_wav
from ..writer import atomic_write

FEATURE_VERSION=2
def frame_to_ms(frame,config): return np.asarray(frame)*config.hop_length/config.sample_rate*1000
def ms_to_frame(ms,config): return np.asarray(ms)*config.sample_rate/(1000*config.hop_length)

def read_experimental_wav(path):
    """Use the common read-only normalization for every estimator."""
    return read_wav(path)

def extract_features(path,config,cache_dir=None):
    path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    signature=json.dumps(dict(version=FEATURE_VERSION,config=asdict(config)),sort_keys=True)
    key=hashlib.sha256((digest+signature).encode()).hexdigest()
    cached=Path(cache_dir)/(key+'.npz') if cache_dir else None
    if cached and cached.exists():
        try:
            with np.load(cached,allow_pickle=False) as z:return {k:z[k] for k in z.files}
        except (ValueError,OSError): pass
    x,sr=read_wav(path)
    if sr!=config.sample_rate:raise ValueError('Sample rate does not match model config')
    spectrum=abs(librosa.stft(x,n_fft=config.n_fft,hop_length=config.hop_length,center=True))**2
    mel=librosa.feature.melspectrogram(S=spectrum,sr=sr,n_fft=config.n_fft,n_mels=config.n_mels,fmin=config.fmin,fmax=config.fmax)
    mel_db=librosa.power_to_db(mel,ref=max(float(mel.max()),1e-10),top_db=config.top_db) if mel.max()>1e-10 else np.full_like(mel,-config.top_db)
    mfcc=librosa.feature.mfcc(S=mel_db,n_mfcc=config.n_mfcc)
    delta=librosa.feature.delta(mfcc,width=5,mode='nearest')
    rms=librosa.feature.rms(y=x,frame_length=config.n_fft,hop_length=config.hop_length,center=True)[0]
    power_db=20*np.log10(np.maximum(rms,1e-8))
    normalized=np.sqrt(spectrum)/(np.sqrt(spectrum).sum(axis=0,keepdims=True)+1e-9)
    flux=np.r_[0,np.linalg.norm(np.diff(normalized,axis=1),axis=0)]
    pitch_size=2048
    frames=librosa.util.frame(np.pad(x,(pitch_size//2,pitch_size//2)),frame_length=pitch_size,hop_length=config.hop_length).T
    frames=frames-frames.mean(axis=1,keepdims=True)
    fft=np.fft.rfft(frames,n=pitch_size*2,axis=1)
    acf=np.fft.irfft(abs(fft)**2,n=pitch_size*2,axis=1)[:,:pitch_size]
    lo=int(sr/config.pitch_max);hi=min(pitch_size-1,int(sr/config.pitch_min))
    lags=np.arange(lo,hi+1)
    ratio=acf[:,lo:hi+1]/(acf[:,0,None]+1e-12)/(1-lags/pitch_size)
    lag=ratio.argmax(axis=1);periodicity=np.clip(ratio[np.arange(len(lag)),lag],0,1)
    voiced=(periodicity>.5)&(power_db>-60)
    f0=np.where(voiced,sr/(lag+lo),0)
    values=dict(mel=mel_db.astype(np.float32),times=frame_to_ms(np.arange(mel.shape[1]),config),
        alignment=np.column_stack((mfcc.T,delta.T,power_db,flux,voiced.astype(float))).astype(np.float32),
        power_db=power_db,flux=flux,voiced=voiced,periodicity=periodicity,f0=f0,
        duration_ms=np.asarray(len(x)/sr*1000),sha256=np.asarray(digest))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('WAV changed during feature extraction')
    if cached:
        stream=io.BytesIO();np.savez_compressed(stream,**values);atomic_write(cached,stream.getvalue(),overwrite=True)
    return values

def independent_slot(features,index,count,vision_config):
    """Alias assignment uses only energy and recording order, never method 1 or oto."""
    power=features['power_db'];times=features['times']
    active=np.flatnonzero(power>max(-60,float(power.max())-vision_config.activity_below_peak_db))
    if len(active)<2:raise ValueError('No effective speech interval')
    start,end=times[active[0]],times[active[-1]]
    period=max(30.,(end-start)/max(count,1))
    center=start+index*period
    radius=vision_config.assignment_radius_beats*period
    return max(0.,center-radius),min(float(features['duration_ms']),center+radius),center
