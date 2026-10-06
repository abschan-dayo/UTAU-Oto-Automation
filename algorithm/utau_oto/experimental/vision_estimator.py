from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .config import Config
from .features import independent_slot,ms_to_frame
from .result import PreutteranceEstimate
from .phonemes import target_index

def peak_confidence(values,step_ms,separation_ms):
    y=np.asarray(values,dtype=float)
    if not len(y):return 0.,{}
    k=int(y.argmax());peak=float(y[k]);radius=max(1,round(separation_ms/step_ms))
    other=y.copy();other[max(0,k-radius):k+radius+1]=0
    second=float(other.max());margin=max(0.,(peak-second)/max(peak,1e-8))
    contrast=max(0.,(peak-float(np.median(y)))/max(peak,1e-8))
    halfwidth=max(1,int(np.count_nonzero(y>=max(.01,.5*peak))))
    sharpness=min(1.,radius/halfwidth)
    score=np.clip(peak*(.4*margin+.3*contrast+.3*sharpness),0,1)
    return float(score),dict(peak=peak,second_peak=second,margin=margin,sharpness=sharpness,contrast=contrast)

class VisionEstimator:
    def __init__(self,model_path,device='auto'):
        self.error=None;self.config=Config();self.cache={};self.sequence_cache={}
        try:
            if not model_path or not Path(model_path).is_file():raise ValueError('Vision model missing')
            import torch
            from .vision_model import build_model,device_for
            self.device=device_for(device)
            checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
            if checkpoint.get('format')!='uto-vision-1':raise ValueError('Unsupported vision model')
            self.config=Config.from_dict(checkpoint['config'])
            self.model=build_model(self.config.vision.architecture).to(self.device);self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()
            self.model_metadata=checkpoint.get('provenance',{})
        except Exception as e:self.error=str(e)

    def heatmap(self,features):
        if self.error:raise RuntimeError(self.error)
        key=str(features['sha256'])
        if key in self.cache:return self.cache[key]
        import torch
        f,v=self.config.features,self.config.vision
        width=max(8,round(float(ms_to_frame(v.window_ms,f))));stride=max(1,round(float(ms_to_frame(v.stride_ms,f))))
        mel=np.clip((features['mel']+f.top_db)/f.top_db,0,1)
        length=mel.shape[1];half=width//2;padded=np.pad(mel,((0,0),(half,width)),mode='edge')
        starts=sorted(set(list(range(0,length,stride))+[length-1]));result=np.zeros(length+width);weight=np.zeros(length+width)
        taper=np.maximum(.1,np.hanning(width))
        with torch.inference_mode():
            for offset in range(0,len(starts),v.batch_size):
                batch=starts[offset:offset+v.batch_size]
                inputs=np.stack([padded[:,s:s+width] for s in batch])[:,None]
                predictions=torch.sigmoid(self.model(torch.as_tensor(inputs,dtype=torch.float32,device=self.device))).cpu().numpy()
                for start,prediction in zip(batch,predictions):
                    left=start-half;lo=max(0,left);hi=min(length,left+width)
                    a=lo-left;b=hi-left
                    result[lo:hi]+=prediction[a:b]*taper[a:b];weight[lo:hi]+=taper[a:b]
        heat=result[:length]/np.maximum(weight[:length],1e-9)
        self.cache={key:heat}
        return heat

    def estimate(self,features,filename,alias,occurrence=0,search_window=None):
        if self.error:return PreutteranceEstimate.unavailable('vision',self.error)
        try:
            index,count=target_index(filename,alias,occurrence)
            if self.config.vision.sequence_decoder=='ordered' and search_window is None:
                from .vision_sequence import decode_sequence
                key=(str(features['sha256']),count)
                if key not in self.sequence_cache:
                    self.sequence_cache={key:decode_sequence(features,self.heatmap(features),count,self.config.vision.sequence_config)}
                positions,sequence=self.sequence_cache[key]
                position=positions[index]
                heat=self.heatmap(features);times=features['times']
                period=sequence.get('period_ms') or self.config.vision.window_ms
                lower=max(0.,position-.25*period);upper=min(float(features['duration_ms']),position+.25*period)
                frames=np.flatnonzero((times>=lower)&(times<=upper))
                if not len(frames):raise ValueError('Empty Vision sequence interval')
                y=heat[frames];frame=int(np.argmin(abs(times-position)))
                step=self.config.features.hop_length/self.config.features.sample_rate*1000
                local,details=peak_confidence(y,step,self.config.vision.peak_separation_ms)
                selected_score=float(heat[frame]);sequence_score=float(sequence['sequence_confidence'])
                confidence=float(np.clip(.4*local+.4*sequence_score+.2*selected_score,0,1))
                peaks,_=find_peaks(y,distance=max(1,round(self.config.vision.peak_separation_ms/step)))
                candidates=sorted(set(peaks.tolist()+[int(np.argmin(abs(times[frames]-position)))]),key=lambda i:y[i],reverse=True)[:8]
                return PreutteranceEstimate('vision',position,confidence,
                    [dict(position_ms=float(times[frames[i]]),score=float(y[i])) for i in candidates],
                    dict(details,sequence=sequence,search_window_ms=[lower,upper],window_source='vision_heatmap_ordered_sequence',
                         confidence_calibrated=False,model=self.model_metadata))
            lower,upper,_=independent_slot(features,index,count,self.config.vision)
            if search_window is not None:lower,upper=search_window
            heat=self.heatmap(features);times=features['times']
            frames=np.flatnonzero((times>=lower)&(times<=upper))
            if not len(frames):raise ValueError('Empty vision search interval')
            y=heat[frames];k=int(y.argmax());frame=int(frames[k])
            step=self.config.features.hop_length/self.config.features.sample_rate*1000
            confidence,details=peak_confidence(y,step,self.config.vision.peak_separation_ms)
            peaks,_=find_peaks(y,distance=max(1,round(self.config.vision.peak_separation_ms/step)))
            peaks=sorted(set(peaks.tolist()+[k]),key=lambda i:y[i],reverse=True)[:8]
            return PreutteranceEstimate('vision',float(times[frame]),confidence,
                [dict(position_ms=float(times[frames[i]]),score=float(y[i])) for i in peaks],
                dict(details,search_window_ms=[lower,upper],window_source='explicit' if search_window else 'independent_energy_and_mora_order',
                     confidence_calibrated=False,model=self.model_metadata))
        except Exception as e:return PreutteranceEstimate.unavailable('vision',e)
