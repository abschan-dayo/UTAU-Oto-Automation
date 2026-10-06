"""Alignment-only voiced onset and learned duration prior; no Vision input."""
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d,maximum_filter1d

def ordered_onset(features,config):
    times=np.asarray(features['times'],float)
    power=np.asarray(features['power_db'],float)
    periodicity=np.asarray(features['periodicity'],float)
    if len(times)<3:raise ValueError('Too few alignment frames')
    step=float(np.median(np.diff(times)))
    active=(power>max(-60.,float(power.max())-config.ordered_onset_power_below_peak_db))&(periodicity>config.ordered_onset_periodicity)
    smooth=max(3,round(config.ordered_onset_smoothing_ms/step)//2*2+1)
    frames=np.flatnonzero(median_filter(active.astype(float),size=smooth)>.5)
    if len(frames)<2:raise ValueError('No stable voiced onset')
    return float(times[frames[0]])

def ordered_slot(features,index,count,period_ms,config,phase_ms=0.):
    if not np.isfinite(period_ms) or period_ms<=0:raise ValueError('Invalid rhythm prior')
    onset=ordered_onset(features,config)
    center=onset+phase_ms+index*period_ms
    radius=config.ordered_radius_beats*period_ms
    return max(0.,center-radius),min(float(features['duration_ms']),center+radius),center


def recording_rhythm(features,filename,count,config):
    """Estimate one rhythm grid from this WAV's numerical acoustic changes.

    The filename supplies only mora order/type. No oto labels or other
    estimator outputs enter this search. Returned phase is relative to the
    first sustained voiced onset used by ``ordered_slot``.
    """
    from ..names import parse_name

    if count<3:raise ValueError('Too few moras for recording-wide rhythm')
    times=np.asarray(features['times'],float)
    power=np.asarray(features['power_db'],float)
    flux=np.asarray(features['flux'],float)
    if (len(times)<3 or power.shape!=times.shape or flux.shape!=times.shape
            or not np.isfinite(times).all() or not np.isfinite(power).all()
            or not np.isfinite(flux).all()):
        raise ValueError('Invalid recording rhythm features')
    step=float(np.median(np.diff(times)))
    if step<=0:raise ValueError('Invalid recording timestamps')
    onset=ordered_onset(features,config)
    active=np.flatnonzero(power>max(-60.,float(power.max())-25.))
    if len(active)<2:raise ValueError('No effective speech end')
    end=float(times[active[-1]])
    moras=parse_name(filename).moras
    if len(moras)!=count:raise ValueError('Mora count changed during alignment')
    indices=np.asarray([i for i,m in enumerate(moras)
                        if i>0 and m.kind not in ('vowel','moraic_nasal')],int)
    if len(indices)<2:indices=np.arange(1,count)

    def normalized(values):
        lo,hi=np.quantile(values,[.4,.98])
        return np.clip((values-lo)/max(float(hi-lo),1e-9),0,1.5)

    change=gaussian_filter1d(flux,2.)
    rise=np.maximum(0.,np.gradient(gaussian_filter1d(power,2.)))
    signal=.75*normalized(change)+.25*normalized(rise)
    window=max(3,round(config.ordered_novelty_window_ms/step)//2*2+1)
    signal=maximum_filter1d(signal,size=window)
    phases=np.arange(-config.ordered_phase_search_ms,
                     config.ordered_phase_search_ms+config.ordered_phase_step_ms*.5,
                     config.ordered_phase_step_ms)
    def search(tail_max,tail_center,tail_std,tail_penalty):
        low=max(config.ordered_period_min_ms,(end-onset)/(count-1+tail_max))
        high=min(config.ordered_period_max_ms,(end-onset)/(count-1+config.ordered_tail_min_beats))
        if high<low:return None,None
        periods=np.arange(low,high+config.ordered_period_step_ms*.5,config.ordered_period_step_ms)
        best=None;second=None
        for period in periods:
            for phase in phases:
                centers=onset+phase+indices*period
                if centers[0]<times[0] or centers[-1]>end-30.:continue
                tail=(end-(onset+phase+(count-1)*period))/period
                score=float(np.mean(np.interp(centers,times,signal))
                            -tail_penalty*((tail-tail_center)/tail_std)**2
                            -config.ordered_phase_penalty*(phase/config.ordered_phase_search_ms)**2)
                if best is None or score>best[0]:
                    second=best;best=(score,float(period),float(phase),float(tail))
                elif second is None or score>second[0]:
                    second=(score,float(period),float(phase),float(tail))
        return best,second
    best,second=search(config.ordered_tail_max_beats,config.ordered_tail_center_beats,
                       config.ordered_tail_std_beats,config.ordered_tail_penalty)
    used_wide_tail=False
    if best is None or best[0]<config.ordered_rhythm_min_score:
        alternative,alternative_second=search(config.ordered_fallback_tail_max_beats,
            config.ordered_fallback_tail_center_beats,config.ordered_fallback_tail_std_beats,
            config.ordered_fallback_tail_penalty)
        if alternative is not None and (best is None or alternative[0]>best[0]+config.ordered_fallback_min_gain):
            best,second=alternative,alternative_second
            used_wide_tail=True
    if best is None:raise ValueError('No valid recording-wide rhythm grid')
    return dict(score=best[0],period_ms=best[1],phase_ms=best[2],tail_beats=best[3],
                second_score=second[0] if second else None,onset_ms=onset,speech_end_ms=end,
                used_wide_tail=used_wide_tail)

def acoustic_phase_shift(features,filename,period_ms,learned_phase_ms,config):
    """Find a recording-wide phase from numerical acoustic changes alone."""
    from ..names import parse_name
    moras=parse_name(filename).moras
    indices=np.asarray([i for i,m in enumerate(moras) if i>0 and m.kind not in ('vowel','moraic_nasal')])
    if len(indices)<2:return 0.
    times=np.asarray(features['times'],float)
    step=float(np.median(np.diff(times)))
    change=gaussian_filter1d(np.asarray(features['flux'],float),15/step)
    change/=max(float(np.quantile(change,.98)),1e-6)
    power=np.asarray(features['power_db'],float)
    rise=np.maximum(0.,np.gradient(gaussian_filter1d(power,10/step)))
    rise/=max(float(np.quantile(rise,.98)),1e-6)
    base=ordered_onset(features,config)+learned_phase_ms+indices*period_ms
    shifts=np.arange(-config.ordered_phase_search_ms,config.ordered_phase_search_ms+2.5,5.)
    scores=[]
    weight=config.ordered_phase_power_weight
    for shift in shifts:
        centers=base+shift
        signal=(1-weight)*np.interp(centers,times,change)+weight*np.interp(centers,times,rise)
        scores.append(float(np.mean(signal)-config.ordered_phase_prior_weight*(shift/config.ordered_phase_search_ms)**2))
    return float(shifts[int(np.argmax(scores))])
