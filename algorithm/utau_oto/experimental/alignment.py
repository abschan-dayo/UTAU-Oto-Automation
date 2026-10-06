"""Diagonal Gaussian phoneme templates and two-best semi-Markov alignment."""
from pathlib import Path
import json
import numpy as np
from .config import Config
from .features import independent_slot
from .phonemes import parse_alias,target_index
from .result import PreutteranceEstimate

def two_best_paths(emission,states,step_ms,config,boundary_range=None,boundary_center_frame=None):
    """Exact top two segmentations with minimum/maximum state durations."""
    a=config;length=len(emission);n_states=len(states)
    prefix=np.vstack([np.zeros(n_states),np.cumsum(emission,axis=0)])
    costs=np.full((n_states+1,length+1,2),np.inf);costs[0,0,0]=0
    back={}
    for s,state in enumerate(states,1):
        consonant=state.startswith('c:')
        minimum=max(1,int(np.ceil((a.consonant_min_ms if consonant else a.vowel_min_ms)/step_ms)))
        maximum=max(minimum,int(np.floor((a.consonant_max_ms if consonant else a.vowel_max_ms)/step_ms)))
        mu=a.consonant_mean_ms if consonant else a.vowel_mean_ms
        sigma=a.consonant_std_ms if consonant else a.vowel_std_ms
        for end in range(1,length+1):
            if s==1 and boundary_range is not None and not boundary_range[0]<=end<=boundary_range[1]:continue
            starts=np.arange(max(0,end-maximum),end-minimum+1)
            if not len(starts):continue
            duration=(end-starts)*step_ms
            segment=prefix[end,s-1]-prefix[starts,s-1]+a.duration_weight*.5*((duration-mu)/max(sigma,1e-6))**2
            if s==1 and boundary_center_frame is not None:
                deviation=(end-boundary_center_frame)*step_ms/max(a.ordered_boundary_prior_std_ms,1e-6)
                segment+=a.ordered_boundary_prior_weight*.5*deviation**2
            options=costs[s-1,starts,:]+segment[:,None]
            flat=options.ravel();valid=np.flatnonzero(np.isfinite(flat))
            if not len(valid):continue
            best=valid[np.argsort(flat[valid],kind='stable')[:2]]
            for rank,idx in enumerate(best):
                costs[s,end,rank]=flat[idx];back[(s,end,rank)]=(int(starts[idx//2]),int(idx%2))
    results=[]
    for rank in (0,1):
        cost=costs[n_states,length,rank]
        if not np.isfinite(cost):continue
        end=length;segments=[];r=rank
        for s in range(n_states,0,-1):
            start,r=back[(s,end,r)];segments.append((start,end));end=start
        results.append((float(cost),list(reversed(segments))))
    return results

class AlignmentEstimator:
    def __init__(self,template_path,apply_calibration=True):
        self.error=None;self.config=Config();self.phase_cache={};self.rhythm_cache={};self.template_path=template_path
        self.apply_calibration=apply_calibration
        try:
            if not template_path or not Path(template_path).is_file():raise ValueError('Alignment templates missing')
            with np.load(template_path,allow_pickle=False) as z:
                self.metadata=json.loads(str(z['metadata']));self.config=Config.from_dict(self.metadata['config'])
                if self.metadata.get('format')!='uto-alignment-1':raise ValueError('Unsupported alignment template format')
                self.names=list(z['names']);self.means=z['means'];self.variances=z['variances']
                self.mean=z['normalization_mean'];self.scale=z['normalization_scale']
        except Exception as e:self.error=str(e)

    def estimate(self,features,filename,alias,occurrence=0):
        if self.error:return PreutteranceEstimate.unavailable('alignment',self.error)
        try:
            phones=parse_alias(alias);states=phones['states'];index,count=target_index(filename,alias,occurrence)
            if any(s not in self.names for s in states):raise ValueError('Missing phoneme template: '+str([s for s in states if s not in self.names]))
            rhythm_info=None;rhythm_fallback=None
            if self.config.alignment.slot_mode=='ordered':
                from .alignment_timing import ordered_slot,acoustic_phase_shift,recording_rhythm
                a=self.config.alignment
                phase_shift=0.
                if a.ordered_rhythm_mode=='per_wav':
                    key=(str(features['sha256']),filename)
                    if key not in self.rhythm_cache:
                        try:self.rhythm_cache[key]=recording_rhythm(features,filename,count,a)
                        except ValueError as exc:self.rhythm_cache[key]=exc
                    rhythm_info=self.rhythm_cache[key]
                    if isinstance(rhythm_info,ValueError):
                        rhythm_fallback=str(rhythm_info);rhythm_info=None
                        lo,hi,center=independent_slot(features,index,count,self.config.vision)
                    else:
                        lo,hi,center=ordered_slot(features,index,count,rhythm_info['period_ms'],a,rhythm_info['phase_ms'])
                else:
                    if a.ordered_phase_mode=='acoustic':
                        key=(str(features['sha256']),filename)
                        if key not in self.phase_cache:
                            self.phase_cache[key]=acoustic_phase_shift(features,filename,self.metadata['rhythm_prior_ms'],
                                self.metadata.get('rhythm_phase_ms',0.),a)
                        phase_shift=self.phase_cache[key]
                    lo,hi,center=ordered_slot(features,index,count,self.metadata['rhythm_prior_ms'],a,
                        self.metadata.get('rhythm_phase_ms',0.)+a.ordered_phase_correction_ms+phase_shift)
            else:lo,hi,center=independent_slot(features,index,count,self.config.vision)
            a=self.config.alignment
            frames=np.flatnonzero((features['times']>=max(0,lo-a.context_before_ms))&(features['times']<=min(float(features['duration_ms']),hi+a.context_after_ms)))
            if len(frames)<len(states):raise ValueError('Too few alignment frames')
            times=features['times'][frames];x=(features['alignment'][frames]-self.mean)/self.scale
            indices=[self.names.index(s) for s in states]
            emission=np.stack([(.5*((x-self.means[i])**2/self.variances[i]+np.log(self.variances[i]))).mean(axis=1) for i in indices],axis=1)
            # Boundary constraint uses recording order only, not a peak or existing estimate.
            step=self.config.features.hop_length/self.config.features.sample_rate*1000
            paths=two_best_paths(emission,states,step,a,(int(np.searchsorted(times,lo)),int(np.searchsorted(times,hi))),
                (center-times[0])/step if a.slot_mode=='ordered' else None)
            if not paths:raise ValueError('No path satisfies duration priors')
            best,segments=paths[0];boundary=segments[1][0];position=float(times[boundary])
            second=paths[1][0] if len(paths)>1 else None
            gap=0. if second is None else max(0.,second-best)
            margin=1-np.exp(-gap/a.gap_scale)
            fit=float(np.exp(-max(0,best/len(frames))))
            dur=[]
            for state,(start,end) in zip(states,segments):
                mu=a.consonant_mean_ms if state.startswith('c:') else a.vowel_mean_ms
                sigma=a.consonant_std_ms if state.startswith('c:') else a.vowel_std_ms
                dur.append(np.exp(-.5*(((end-start)*step-mu)/sigma)**2))
            before=x[max(0,boundary-4):boundary];after=x[boundary:min(len(x),boundary+4)]
            contrast=float(np.tanh(np.linalg.norm(before.mean(0)-after.mean(0))/np.sqrt(x.shape[1]))) if len(before) and len(after) else 0.
            confidence=float(np.clip(.35*fit+.25*margin+.2*np.mean(dur)+.2*contrast,0,1))
            if not phones['consonant'] and phones['previous']==phones['vowel']:confidence*=.4
            if not lo<=position<=hi:confidence*=.4
            if rhythm_info is not None:
                confidence*=float(np.clip(.3+.7*max(0.,rhythm_info['score']),.1,1.))
                if rhythm_info['used_wide_tail']:confidence*=.85
            elif rhythm_fallback is not None:confidence*=.3
            candidates=[dict(position_ms=float(times[p[1][1][0]]),path_cost=p[0]) for p in paths]
            estimate=PreutteranceEstimate('alignment',position,confidence,candidates,
                dict(best_path_cost=best,second_best_path_cost=second,path_margin=float(margin),template_fit=fit,
                    duration_fit=float(np.mean(dur)),boundary_contrast=contrast,confidence_calibrated=False,
                    acoustic_phase_shift_ms=phase_shift if a.slot_mode=='ordered' and a.ordered_rhythm_mode=='global' else None,
                    recording_rhythm=rhythm_info,recording_rhythm_fallback=rhythm_fallback,
                    search_window_ms=[float(times[0]),float(times[-1])],
                    states=[dict(phoneme=s,start_ms=float(times[start]),end_ms=float(times[min(end,len(times)-1)])) for s,(start,end) in zip(states,segments)]))
            if not self.apply_calibration:return estimate
            from .alignment_calibration import calibrate
            return calibrate(estimate,features,index,count,phones,self.template_path)
        except Exception as e:return PreutteranceEstimate.unavailable('alignment',e)
