"""Sustained spectral boundaries, with explicit competing-candidate diagnostics."""
import numpy as np
from scipy.signal import find_peaks


def estimate(f, lower, upper, anchor, config, cap=100., target_region=None, require_release=False):
    n=len(f.times)
    lower=max(0,int(lower));upper=min(n-1,max(lower,int(upper)))
    # Normalized spectral shape supplies the primary evidence. Power is
    # only an audibility / release check, never sufficient on its own.
    shape=np.asarray(f.bands,dtype=float)
    width=max(4,config.stable_frames*2)
    delta=np.zeros(n)
    for k in range(lower,upper+1):
        before=np.median(shape[max(0,k-width):max(1,k)],axis=0)
        after=np.median(shape[k:min(n,k+width)],axis=0)
        delta[k]=np.linalg.norm(after-before)
    local=delta[lower:upper+1]
    peaks,_=find_peaks(local,prominence=.025,distance=config.candidate_distance)
    pool=set((peaks+lower).tolist())
    pool.update((lower,upper,int(np.clip(anchor,lower,upper)),lower+int(np.argmax(local))))
    target=None if target_region is None else np.median(shape[target_region[0]:target_region[1]+1],axis=0)
    for peak in list(pool):
        if delta[peak]<.10: continue
        edge=peak
        while edge>lower and delta[edge-1]>=.60*delta[peak]: edge-=1
        pool.add(edge)
    records=[]
    for k in sorted(pool):
        # A longer preceding median prevents the end of a short glitch from
        # masquerading as a new sustained state.
        before=np.median(shape[max(0,k-2*width):max(1,k)],axis=0)
        following=shape[k:min(n,k+2*width)]
        after=np.median(following[:width],axis=0)
        separation=float(np.linalg.norm(after-before))
        d_before=np.linalg.norm(following-before,axis=1)
        d_after=np.linalg.norm(following-after,axis=1)
        changed=(d_before>max(.045,.35*separation)) & (d_after<d_before)
        persistence=float(np.mean(changed)) if len(changed) else 0.
        tail=float(np.mean(changed[width:])) if len(changed)>width else 0.
        returning=1-tail
        variation=float(np.median(np.linalg.norm(following-after,axis=1)))
        stability=float(np.exp(-3*variation))
        spectral=float(np.clip(separation/.35,0,1))
        p0=float(np.median(f.power_db[max(0,k-width):max(1,k)]))
        p1=float(np.median(f.power_db[k:min(n,k+width)]))
        rms=float(np.clip(abs(p1-p0)/8,0,1))
        audible=float(np.mean(f.power_db[k:min(n,k+2*width)]>f.noise_floor_db+10))
        progress=1.
        if target is not None:
            old_distance=float(np.linalg.norm(before-target))
            new_distance=float(np.linalg.norm(after-target))
            progress=float(np.clip((old_distance-new_distance)/max(old_distance,.05),0,1))
        decay=float(np.clip((p0-p1)/6,0,1)) if require_release else 0.
        lateness=(k-lower)/max(upper-lower,1)
        score=.50*spectral+.30*persistence+.10*stability+.05*rms-.25*returning-.10*lateness-.45*(1-audible)-.20*(1-progress)-.30*decay
        records.append(dict(index=k,score=float(score),spectral=spectral,
            structure_difference=separation,persistence=persistence,stability=stability,
            return_penalty=returning,rms=rms,late_penalty=lateness,audible=audible,target_progress=progress,decay_penalty=decay))
    # Do not let a later larger lobe replace an already clear sustained change.
    credible=[r for r in records if r['spectral']>=.45 and r['persistence']>=.70 and r['return_penalty']<=.25 and r['audible']>=.8 and r['target_progress']>=.10 and r['decay_penalty']<.5]
    if credible:
        strongest=max(r['score'] for r in credible)
        eligible=[r for r in credible if r['score']>=strongest-.12]
        winner=min(eligible,key=lambda r:r['index'])
    else:
        winner=max(records,key=lambda r:(r['score'],-abs(r['index']-anchor),-r['index']))
    others=sorted((r for r in records if r is not winner),key=lambda r:(-r['score'],r['index']))
    competitors=[r for r in others if abs(r['index']-winner['index'])>=config.candidate_distance]
    margin=winner['score']-competitors[0]['score'] if competitors else 0.
    confidence=100*(.45*winner['spectral']+.30*winner['persistence']+.15*max(0,min(1,margin/.2))+.10*winner['rms'])
    confidence*=1-.5*winner['return_penalty']
    if not credible: confidence=min(confidence,40.)
    if margin<.04: confidence=min(confidence,49.)
    if winner['index'] in (lower,upper): confidence=min(confidence,45.)
    confidence=round(float(np.clip(confidence,0,cap)),2)
    reason=['スペクトル構造変化明瞭' if winner['spectral']>=.45 else '構造変化が弱い',
            '持続性高' if winner['persistence']>=.70 else '持続性不足']
    if margin<.04: reason.append('第2候補と競合 / 要確認')
    if winner['rms']>=.5: reason.append('RMS変化あり')
    candidates=[]
    for r in [winner]+others[:11]:
        candidates.append({key:round(float(value),6) for key,value in r.items() if key!='index'} |
            dict(absolute_ms=round(float(f.times[r['index']]),3),selected=r is winner))
    return winner['index'],dict(label='先行発声',confidence=confidence,
        selected_absolute_ms=round(float(f.times[winner['index']]),3),candidates=candidates,
        candidate_margin=round(float(margin),6),reason=' / '.join(reason),
        placement_method='最初の持続するスペクトル構造変化',
        warning=confidence<config.line_warning_percent)
