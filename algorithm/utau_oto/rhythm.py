"""Per-recording tempo grid; one recorded mora is treated as one pulse."""
import numpy as np


def repeated_vowel(moras, i):
    return bool(i and moras[i].kind=='vowel' and moras[i-1].vowel==moras[i].vowel)


def fit_grid(times, moras, features=None, evidence=None, short_period=None):
    times=np.asarray(times,dtype=float)
    ids=np.array([i for i in range(len(times)) if not repeated_vowel(moras,i)])
    if len(times)<=2 and short_period is not None:
        grid=times[0]+np.arange(len(times))*short_period
        return grid,dict(bpm=round(60000/short_period,3),period_ms=float(short_period),phase_ms=float(times[0]),
            reference_indices=[0],method='短い録音: 同フォルダの長い録音から拍間隔を補助',beat_convention='1音素=1拍')
    if features is not None and len(times)>=3:
        # A missed consonant creates a double gap; it must not stretch every
        # following beat. Seed tempo from ordinary adjacent intervals, then
        # score a complete periodic lattice against the acoustic evidence.
        gaps=np.diff(times)
        base=float(np.median(gaps))
        periods=np.linspace(.75*base,1.2*base,91)
        best=None
        for period in periods:
            for phase in np.linspace(times[0]-.15*period,times[0]+.45*period,25):
                grid=phase+np.arange(len(times))*period
                if grid[-1]>times[-1]+.1*period: continue
                support=[]
                for i,t in enumerate(grid):
                    if repeated_vowel(moras,i):continue
                    mask=abs(features.times-t)<=.10*period
                    support.append(float(np.max(evidence[i][mask]*(1-.5*abs(features.times[mask]-t)/(.10*period)))) if np.any(mask) else 0.)
                score=float(np.mean(support))-.10*abs(period/base-1)-.05*abs(phase-times[0])/period
                candidate=(score,-abs(period-base),-phase,period,phase)
                if best is None or candidate>best:best=candidate
        if best is not None:
            _,_,_,period,phase=best
            grid=phase+np.arange(len(times))*period
            return grid,dict(bpm=round(60000/period,3),period_ms=round(period,3),phase_ms=round(phase,3),
                reference_indices=ids.tolist(),method='周期格子全体と音響境界を照合',beat_convention='1音素=1拍')
    if len(ids)<2:
        period=float(np.median(np.diff(times))) if len(times)>1 else 300.
        return times.copy(),dict(bpm=60000/max(period,1.),period_ms=period,
            phase_ms=float(times[0]),reference_indices=ids.tolist(),method='拍の根拠不足: 初期整列を維持')
    slopes=[(times[j]-times[i])/(j-i) for a,i in enumerate(ids) for j in ids[a+1:]]
    # Median pairwise slope resists an isolated displaced boundary.
    period=max(1.,float(np.median(slopes)))
    phase=float(np.median(times[ids]-ids*period))
    residual=times[ids]-(phase+ids*period)
    inliers=ids[np.abs(residual)<=max(.25*period,15.)]
    if len(inliers)>=2:
        slopes=[(times[j]-times[i])/(j-i) for a,i in enumerate(inliers) for j in inliers[a+1:]]
        period=max(1.,float(np.median(slopes)))
        phase=float(np.median(times[inliers]-inliers*period))
    else:
        inliers=ids
    grid=phase+np.arange(len(times))*period
    return grid,dict(bpm=round(60000/period,3),period_ms=round(period,3),phase_ms=round(phase,3),
        reference_indices=inliers.tolist(),residual_ms=round(float(np.median(abs(times[inliers]-grid[inliers]))),3),
        method='同母音連続以外の境界から拍間隔を推定',beat_convention='1音素=1拍（倍テンポ・半テンポの区別は未確定）')


def interpolate_repeats(positions, moras, period):
    result=np.asarray(positions,dtype=float).copy()
    refs=[i for i in range(len(result)) if not repeated_vowel(moras,i)]
    for i in range(len(result)):
        if not repeated_vowel(moras,i): continue
        left=[j for j in refs if j<i];right=[j for j in refs if j>i]
        if left and right:
            a,b=left[-1],right[0]
            result[i]=result[a]+(result[b]-result[a])*(i-a)/(b-a)
        elif left: result[i]=result[left[-1]]+(i-left[-1])*period
        elif right: result[i]=result[right[0]]-(right[0]-i)*period
    return result


def seed_lines(grid, period, duration, moras=None):
    result=[]
    for i,pre in enumerate(grid):
        pre=float(np.clip(pre,.003,max(.003,duration-.003)))
        lead=.15 if i==0 and moras is not None and moras[i].kind=="vowel" else .39
        off=max(0.,pre-lead*period)
        fixed=min(duration-.002,pre+.08*period)
        end=min(duration,pre+.74*period)
        result.append(dict(offset=off,overlap=off+(pre-off)/3,preutterance=pre,
            consonant=max(pre,fixed),cutoff=max(pre+.001,end)))
    return result
