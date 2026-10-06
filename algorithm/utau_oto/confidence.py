"""Per-line self-diagnostics; no reference labels or probabilities of truth."""
import numpy as np
from scipy.signal import find_peaks

LABELS = dict(offset='左ブランク',overlap='オーバーラップ',preutterance='先行発声',
              consonant='固定範囲',cutoff='右ブランク')


def harmonic_mean(values):
    values = [float(v) for v in values]
    if not values or any(v <= 0 for v in values):
        return 0.
    return len(values)/sum(1/v for v in values)


def choose(f, field, lower, upper, preferred, components, config, cap=100., weights=None):
    """Evaluate independent evidence at multiple separated local candidates."""
    lower=max(0,int(lower));upper=min(len(f.times)-1,max(lower,int(upper)))
    preferred=int(np.clip(preferred,lower,upper))
    spectral=np.clip(components['spectral'],0,1)
    rms=np.clip(components['rms'],0,1)
    stability=np.clip(components['stability'],0,1)
    sw,rw,tw=weights or (.60,.25,.15)
    base=sw*spectral+rw*rms+tw*stability
    span=max(upper-lower,1)
    prior=1-np.minimum(1,np.abs(np.arange(len(base))-preferred)/span)
    ranking=.85*base+.15*prior
    peaks,_=find_peaks(ranking[lower:upper+1],distance=config.candidate_distance)
    pool=set((peaks+lower).tolist()+[lower,upper,preferred])
    if upper-lower>2:
        pool.add((lower+upper)//2)
    ordered=sorted(pool,key=lambda j:(-float(ranking[j]),abs(j-preferred),j))
    winner=ordered[0]
    alternatives=[j for j in ordered[1:] if abs(j-winner)>=config.candidate_distance]
    # A broad, indistinguishable plateau correctly produces a small margin.
    margin=float(ranking[winner]-ranking[alternatives[0]]) if alternatives else 0.
    agreement=1-abs(float(spectral[winner]-rms[winner]))
    certainty=100*(.5*float(base[winner])+.25*agreement+.25*min(1,max(0,margin)*4))
    if len(alternatives)==0: certainty=min(certainty,45.)
    if float(rms[winner] if rw>sw else spectral[winner])<.2: certainty=min(certainty,40.)
    # F0 is only an agreement bonus. It does not change the selected position.
    if f.f0[winner]>0 and f.world_f0[winner]>0 and abs(np.log2(f.f0[winner]/f.world_f0[winner]))<.08:
        certainty+=3
    certainty=round(float(np.clip(certainty,0,cap)),2)
    reasons=[]
    reasons.append('スペクトル支持' if spectral[winner]>=.5 else 'スペクトル根拠弱')
    if rms[winner]>=.5: reasons.append('RMS一致')
    if margin<.08: reasons.append('候補差小')
    candidates=[]
    for j in [winner]+[j for j in ordered if j!=winner][:7]:
        candidates.append(dict(absolute_ms=round(float(f.times[j]),3),score=round(float(ranking[j]),6),
            spectral=round(float(spectral[j]),4),rms=round(float(rms[j]),4),
            stability=round(float(stability[j]),4),selected=j==winner))
    return winner,dict(label=LABELS[field],confidence=certainty,
        selected_absolute_ms=round(float(f.times[winner]),3),candidates=candidates,
        candidate_margin=round(margin,6),reason=' / '.join(reasons),
        warning=certainty<config.line_warning_percent)


def summarize(lines,config):
    overall=round(harmonic_mean(x['confidence'] for x in lines.values()),2)
    weakest=sorted(lines,key=lambda k:(lines[k]['confidence'],list(LABELS).index(k)))[:2] if overall<config.overall_review_percent else []
    return overall,[dict(field=k,label=LABELS[k],confidence=lines[k]['confidence']) for k in weakest]
