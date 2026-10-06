"""Independent order-constrained assignment of Vision heatmap boundaries.

Only acoustic features, CNN heat, and the filename mora count enter inference.
No existing-estimator output or reference oto positions are used.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


@dataclass(frozen=True)
class SequenceConfig:
    """Serializable independent decoder settings; milliseconds unless specified."""
    voiced_below_peak_db: float = 18.
    power_floor_db: float = -60.
    periodicity_threshold: float = .7
    voiced_smoothing_ms: float = 25.
    fallback_below_peak_db: float = 28.
    heat_quantile: float = .98
    minimum_heat_scale: float = .05
    maximum_normalized_heat: float = 1.5
    peak_separation_ms: float = 35.
    single_search_radius_ms: float = 150.
    single_confidence_cap: float = .3
    minimum_interval_ms: float = 100.
    long_sequence_min_count: int = 4
    period_min_ms: float = 120.
    period_max_ms: float = 1200.
    extra_tail_moras: float = 3.
    short_period_min_ms: float = 100.
    short_period_max_ms: float = 1500.
    short_extra_tail_moras: float = 2.
    phase_before_ms: float = 140.
    phase_after_ms: float = 85.
    phase_step_frames: float = 2.
    search_radius_beats: float = .24
    distance_penalty: float = .85
    missing_peak_penalty: float = .5
    first_mora_weight: float = .5
    phase_penalty: float = .1
    phase_penalty_scale_ms: float = 150.
    alternative_period_ratio: float = .08
    alternative_phase_beats: float = .15
    margin_scale: float = .2
    confidence_fit_weight: float = .7

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Sequence settings must be finite')
        if not (0 < self.periodicity_threshold <= 1 and 0 < self.heat_quantile <= 1
                and 0 < self.search_radius_beats < .5
                and 0 <= self.confidence_fit_weight <= 1 and 0 <= self.single_confidence_cap <= 1):
            raise ValueError('Invalid sequence ratios')
        if not (0 < self.period_min_ms < self.period_max_ms
                and 0 < self.short_period_min_ms < self.short_period_max_ms):
            raise ValueError('Invalid sequence period range')
        positive = ('voiced_below_peak_db', 'voiced_smoothing_ms', 'fallback_below_peak_db',
                    'minimum_heat_scale', 'maximum_normalized_heat', 'peak_separation_ms',
                    'single_search_radius_ms', 'minimum_interval_ms', 'phase_step_frames',
                    'first_mora_weight', 'phase_penalty_scale_ms', 'margin_scale')
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError('Sequence lengths and scales must be positive')
        nonnegative = ('extra_tail_moras', 'short_extra_tail_moras', 'phase_before_ms',
                       'phase_after_ms', 'distance_penalty', 'missing_peak_penalty',
                       'phase_penalty', 'alternative_period_ratio', 'alternative_phase_beats')
        if any(getattr(self, key) < 0 for key in nonnegative) or self.phase_before_ms + self.phase_after_ms <= 0:
            raise ValueError('Invalid sequence penalties or search range')
        if not isinstance(self.long_sequence_min_count, int) or self.long_sequence_min_count < 2:
            raise ValueError('Invalid long sequence threshold')


def _speech_interval(features,config):
    times=np.asarray(features['times'],dtype=float)
    power=np.asarray(features['power_db'],dtype=float)
    periodicity=np.asarray(features['periodicity'],dtype=float)
    voiced=(power>max(config.power_floor_db,float(np.max(power))-config.voiced_below_peak_db))&(periodicity>config.periodicity_threshold)
    step=float(np.median(np.diff(times))) if len(times)>1 else 5.
    # Short unvoiced gaps do not move the onset to the next voiced mora.
    voiced=median_filter(voiced.astype(float),size=max(3,round(config.voiced_smoothing_ms/step)//2*2+1))>.5
    active=np.flatnonzero(voiced)
    if len(active)<2:
        active=np.flatnonzero(power>max(config.power_floor_db,float(np.max(power))-config.fallback_below_peak_db))
    if len(active)<2:raise ValueError('No effective speech interval')
    return float(times[active[0]]),float(times[active[-1]]),step


def decode_sequence(features,heat,count,config=None):
    """Return (ordered absolute positions in ms, diagnostic metadata).

    ``config`` accepts SequenceConfig or a JSON-compatible dictionary. The first
    sustained voiced interval provides only a phase prior: actual boundaries and
    the recording period are selected from the Vision model's heatmap.
    """
    config=SequenceConfig() if config is None else (SequenceConfig(**config) if isinstance(config,dict) else config)
    if not isinstance(config,SequenceConfig):raise ValueError('Expected SequenceConfig')
    times=np.asarray(features['times'],dtype=float)
    heat=np.asarray(heat,dtype=float)
    if not isinstance(count,(int,np.integer)) or count<1 or times.ndim!=1 or heat.ndim!=1 or len(heat)!=len(times) or len(heat)<2:
        raise ValueError('Invalid sequence input')
    if not np.isfinite(heat).all() or np.any(heat<0) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid heatmap or timestamps')
    for key in ('power_db','periodicity'):
        value=np.asarray(features[key])
        if value.shape!=times.shape or not np.isfinite(value).all():raise ValueError('Invalid '+key+' features')
    onset,end,step=_speech_interval(features,config)
    normalized=np.clip(heat/max(float(np.quantile(heat,config.heat_quantile)),config.minimum_heat_scale),0,config.maximum_normalized_heat)
    peaks,_=find_peaks(heat,distance=max(1,round(config.peak_separation_ms/step)))
    peaks=np.unique(np.r_[peaks,int(np.argmax(heat))])
    peak_times=times[peaks]
    if count==1:
        select=np.flatnonzero((peak_times>=onset-config.single_search_radius_ms)&(peak_times<=onset+config.single_search_radius_ms))
        if len(select):
            selected=int(select[np.argmax(normalized[peaks[select]])]);frame=int(peaks[selected])
        else:
            frame=int(np.argmin(abs(times-onset)))
        confidence=config.single_confidence_cap*float(np.clip(heat[frame],0,1))
        return [float(times[frame])],dict(onset_ms=onset,speech_end_ms=end,period_ms=None,sequence_confidence=confidence)
    duration=max(config.minimum_interval_ms,end-onset)
    if count>=config.long_sequence_min_count:
        low=max(config.period_min_ms,duration/(count+config.extra_tail_moras));high=min(config.period_max_ms,duration/(count-1.))
    else:
        low=max(config.short_period_min_ms,duration/(count+config.short_extra_tail_moras));high=min(config.short_period_max_ms,duration/max(1,count-1))
    periods=np.arange(low,high+step,step)
    offsets=np.arange(-config.phase_before_ms,config.phase_after_ms,step*config.phase_step_frames)
    order=np.arange(count)
    best=None;ranked=[]
    for period in periods:
        radius=config.search_radius_beats*period
        for offset in offsets:
            centers=onset+offset+order*period
            if centers[0]<times[0] or centers[-1]>end:continue
            indices=[];scores=[]
            for index,center in enumerate(centers):
                candidates=np.flatnonzero(abs(peak_times-center)<=radius)
                if not len(candidates):
                    frame=int(np.argmin(abs(times-center)));score=float(normalized[frame])-config.missing_peak_penalty
                else:
                    distance=(peak_times[candidates]-center)/period
                    value=normalized[peaks[candidates]]-config.distance_penalty*(distance/config.search_radius_beats)**2
                    chosen=int(candidates[np.argmax(value)]);frame=int(peaks[chosen]);score=float(np.max(value))
                indices.append(frame);scores.append(score)
            # First onset has fewer supervised examples; preserve it as a weak anchor.
            if any(b<=a for a,b in zip(indices,indices[1:])):continue
            weights=np.ones(count);weights[0]=config.first_mora_weight
            score=float(np.average(scores,weights=weights))-config.phase_penalty*(offset/config.phase_penalty_scale_ms)**2
            ranked.append((score,period,offset))
            if best is None or score>best[0]:best=(score,period,offset,indices)
    if best is None:raise ValueError('No valid Vision sequence')
    score,period,offset,indices=best
    alternatives=[s for s,p,o in ranked if abs(p-period)>config.alternative_period_ratio*period or abs(o-offset)>config.alternative_phase_beats*period]
    second=max(alternatives) if alternatives else score
    fit=float(np.clip(score,0,1));margin=float(np.clip((score-second)/config.margin_scale,0,1))
    confidence=config.confidence_fit_weight*fit+(1-config.confidence_fit_weight)*margin
    positions=[float(times[i]) for i in indices]
    return positions,dict(onset_ms=onset,speech_end_ms=end,period_ms=float(period),phase_ms=float(onset+offset),
                          score=float(score),alternative_score=float(second),sequence_confidence=float(confidence))
