"""Monotonic acoustic alignment and image-derived VCV placement rules.

The duration term regularizes alignment; it is not a BPM estimate. Ambiguous
same-vowel sequences cannot be identified reliably without acoustic evidence.
All confidence values are heuristic, never probabilities of correctness.
"""
from dataclasses import asdict
import numpy as np
from .features import smooth, scale
from .names import aliases, hiragana
from scipy.signal import find_peaks
from .config import AnalysisConfig
from .confidence import choose, summarize
from .preutterance import estimate
from .rhythm import fit_grid,interpolate_repeats,repeated_vowel,seed_lines


def evidence(f, kind, previous_kind=None):
    rise = scale(np.maximum(np.gradient(smooth(f.power_db)), 0))
    change = scale(f.flux)
    pitch = np.zeros(len(f.f0))
    ok = (f.f0[1:] > 0) & (f.f0[:-1] > 0)
    pitch[1:] = np.where(ok, np.abs(np.log2(np.maximum(f.f0[1:],1)/np.maximum(f.f0[:-1],1))), 0)
    pitch = scale(smooth(pitch))
    voiced = smooth(f.periodicity)
    onset = scale(np.maximum(np.gradient(voiced), 0))
    if kind == 'moraic_nasal':
        fall = scale(np.maximum(-np.gradient(smooth(f.power_db)), 0))
        return .55*change + .45*fall
    if previous_kind == 'moraic_nasal':
        return .6*change + .4*rise
    if kind in ('vowel', 'glide', 'nasal', 'tap'):
        return .7*change + .3*rise
    frequencies=f.frequencies_hz
    selected=(frequencies>=700)&(frequencies<4500)
    band_power=10*np.log10(np.maximum(np.sum(10**(f.spectrum_db[:,selected]/10),axis=1),1e-12))
    formant_rise=scale(np.maximum(np.gradient(smooth(band_power)),0))
    return .3*change + .2*rise + .5*formant_rise


def align(f, moras):
    n, count = len(f.times), len(moras)
    mask = ((f.power_db > max(-60., float(f.power_db.max())-22.)) &
            (f.flatness < .35) &
            (np.sum(f.bands[:,1:]**2,axis=1) > .08))
    # A click or breath before recording is not the first mora. Require a
    # sustained voiced run, then retain its first frame (no onset delay).
    runs = []
    run = None
    for k, good in enumerate(np.r_[mask, False]):
        if good and run is None:
            run = k
        if not good and run is not None:
            if k-run >= 6:
                runs.extend(range(run,k))
            run = None
    active = np.asarray(runs, dtype=int)
    silent = len(active) == 0 or f.power_db.max() < -65
    first, last = (0, n-1) if silent else (int(active[0]), int(active[-1]))
    if last-first < count*2:
        raise ValueError('録音長に対して音素数が多すぎます。ファイル名とWAVを確認してください')
    spacing = (last-first)/count
    minimum = max(2, min(12, int(spacing*.25)))
    # DP over each frame and mora, allowing strongly uneven durations.
    scores = [evidence(f, m.kind, moras[i-1].kind if i else None) for i,m in enumerate(moras)]
    for i in range(1,count):
        if moras[i].kind=="vowel" and moras[i-1].vowel==moras[i].vowel:
            scores[i]=np.zeros(n)  # No identifiable transition for a repeated vowel.
    dp = np.full((count, n), -np.inf)
    back = np.full((count, n), -1, dtype=int)
    dp[0, first] = 0.
    max_gap = max(minimum+1, int(spacing*3.5))
    for i in range(1, count):
        for j in range(first+i*minimum, last-(count-i-1)*minimum):
            ks = np.arange(max(first+(i-1)*minimum, j-max_gap), j-minimum+1)
            if not len(ks):
                continue
            # Small regularizer prevents packing all boundaries into one burst.
            duration_penalty = .60*np.log(np.maximum((j-ks)/spacing, 1e-6))**2
            values = dp[i-1, ks] - duration_penalty
            k = int(np.argmax(values))
            dp[i,j] = values[k] + scores[i][j]
            back[i,j] = ks[k]
    terminal = dp[-1] - .60*np.log(np.maximum((last-np.arange(n))/spacing, 1e-6))**2
    j = int(np.argmax(terminal))
    if not np.isfinite(terminal[j]):
        raise ValueError('音素境界の単調整列に失敗しました')
    anchors = [j]
    for i in range(count-1, 0, -1):
        j = int(back[i,j]); anchors.append(j)
    return list(reversed(anchors)), last, scores, silent


def stable_region(f, start, end):
    """Find the strongest sustained voiced/stable run inside this mora."""
    end = max(start+1, end)
    ids = np.arange(start, end)
    local_power = f.power_db[ids]
    usable = ((local_power >= np.max(local_power)-9) &
              (f.flatness[ids] < .35) & (f.flux[ids] < .3))
    # Fill isolated grid holes, but never bridge long interruptions.
    if len(usable) > 2:
        usable[1:-1] |= usable[:-2] & usable[2:]
    runs = []
    run = None
    for p, good in enumerate(np.r_[usable, False]):
        if good and run is None:
            run = p
        if not good and run is not None:
            runs.append((run, p)); run = None
    if runs:
        a,b = max(runs, key=lambda r: r[1]-r[0])
        return start+a, start+b-1, b-a >= 4
    # Still return a candidate; the caller makes review mandatory.
    k = int(np.argmax(local_power))
    return start+k, min(end-1, start+k+1), False


def vowel_transition(f, anchor, previous_region, target_region):
    """Find sustained target-vowel evidence after a spectral-change peak."""
    ps,pe,previous_ok = previous_region
    ts,te,target_ok = target_region
    if not previous_ok or not target_ok or te-anchor < 4:
        return anchor, max(anchor,ts), False
    before = np.median(f.bands[ps:pe+1],axis=0)
    after = np.median(f.bands[ts:te+1],axis=0)
    if np.linalg.norm(after-before) < .08:
        return anchor,max(anchor,ts),False
    d0 = np.linalg.norm(f.bands-before,axis=1)
    d1 = np.linalg.norm(f.bands-after,axis=1)
    progress = smooth(d0/np.maximum(d0+d1,1e-9),3)
    power_floor = float(np.median(f.power_db[ts:te+1]))-3.
    arrived = ((progress >= .65) & (f.power_db >= power_floor) & (f.flatness < .35))
    # Do not jump past the early half of the target's stable region.
    limit = min(te-2, ts+(te-ts)//2)
    for k in range(anchor,limit+1):
        if np.all(arrived[k:k+3]):
            fixed=max(k,ts)
            for j in range(fixed,limit+1):
                if np.all(progress[j:j+3]>=.8) and np.all(f.flux[j:j+3]<.3):
                    fixed=j;break
            return k,fixed,True
    return anchor,max(anchor,ts),False


def spectral_onsets(f, lower, upper, baseline_region):
    """Score the leading edges of sustained spectral-change lobes.

    Each edge is derived from its own local floor and peak, rather than a
    constant millisecond shift or the loudest point of the target vowel.
    """
    lower=max(0,int(lower));upper=min(len(f.times)-1,int(upper))
    curve=np.zeros(len(f.times))
    bs,be=baseline_region
    baseline=float(np.median(f.flux[max(0,bs):max(bs+1,be)]))
    local=f.flux[lower:upper+1]
    peaks,_=find_peaks(local,prominence=.035,distance=3)
    if len(local) and not len(peaks): peaks=np.array([int(np.argmax(local))])
    evidence=[]
    for peak in peaks+lower:
        height=float(f.flux[peak]-baseline)
        if height<.06: continue
        threshold=baseline+.4*height
        edge=int(peak)
        while edge>lower and f.flux[edge-1]>=threshold: edge-=1
        if edge==lower or peak-edge<2: continue
        weight=height
        evidence.append((edge,weight))
    if not evidence: return curve,None
    maximum=max(w for _,w in evidence)
    for edge,weight in evidence:
        # A narrow support region preserves nearby competing candidates.
        for delta,factor in [(-1,.7),(0,1.),(1,.7)]:
            k=edge+delta
            if 0<=k<len(curve): curve[k]=max(curve[k],factor*weight/maximum)
    preferred=max(evidence,key=lambda item:(item[1],-item[0]))[0]
    return curve,preferred


def power_onsets(f, lower, upper):
    """Find an audible envelope rise, rejecting flat power and single-frame spikes."""
    lower=max(3,int(lower));upper=min(len(f.times)-4,int(upper))
    curve=np.zeros(len(f.times))
    power=smooth(f.power_db,3)
    slope=np.gradient(power)
    peaks,_=find_peaks(slope[lower:upper+1],prominence=.15,distance=3)
    candidates=[]
    for peak in peaks+lower:
        if slope[peak]<.35: continue
        before=float(np.median(power[peak-3:peak]))
        after=float(np.median(power[peak+1:peak+4]))
        if after-before<3.: continue
        # The waveform envelope must also grow; spectral movement alone is
        # deliberately insufficient for the power-first branch.
        old=float(np.median(f.peak[peak-3:peak]))
        new=float(np.median(f.peak[peak+1:peak+4]))
        if new<=max(old*1.1,old+.002): continue
        edge=int(peak)
        while edge>lower and slope[edge-1]>=.4*slope[peak]: edge-=1
        candidates.append(edge)
    if not candidates: return curve,None
    # First sustained rise, not a later louder part of the same vowel.
    preferred=min(candidates)
    for edge in candidates:
        for delta,factor in [(-1,.7),(0,1.),(1,.7)]:
            curve[edge+delta]=max(curve[edge+delta],factor)
    return curve,preferred


def analyze(parsed, f, review_threshold=.6, config=None):
    config=config or AnalysisConfig(overall_review_percent=100*review_threshold)
    anchors,last,scores,silent=align(f,parsed.moras)
    coarse_anchors=anchors.copy()
    grid,tempo=fit_grid([f.times[a] for a in anchors],parsed.moras,features=f,evidence=scores,short_period=config.short_recording_period_ms)
    grid=np.clip(grid,0,f.times[last])
    seeds=seed_lines(grid,tempo['period_ms'],f.duration_ms,parsed.moras)
    beat_frames=max(3,int(round(tempo['period_ms']/np.median(np.diff(f.times)))))
    anchors=[int(np.argmin(abs(f.times-t))) for t in grid]
    # Keep every mora's search interval nonempty, even for a very short WAV.
    for j in range(len(anchors)):
        anchors[j]=max(j*2,min(anchors[j],last-2*(len(anchors)-1-j)))
        if j: anchors[j]=max(anchors[j],anchors[j-1]+2)
    regions=[stable_region(f,a,anchors[i+1] if i+1<len(anchors) else last+1) for i,a in enumerate(anchors)]
    spectral_stable=np.exp(-6*f.flux)*(1-np.clip(f.flatness,0,1))
    energy=np.clip((f.power_db-f.noise_floor_db-6)/20,0,1)
    stable=spectral_stable*energy
    rising=np.clip(scale(np.maximum(np.gradient(smooth(f.power_db)),0)),0,1)
    falling=np.clip(scale(np.maximum(-np.gradient(smooth(f.power_db)),0)),0,1)
    flux=np.clip(scale(f.flux),0,1)
    estimates=[]
    for i,anchor in enumerate(anchors):
        ts,te,_=regions[i]
        lo=max(1,anchor-int(config.bpm_max_shift_beats*beat_frames))
        hi=min(last-1,anchor+int(config.bpm_max_shift_beats*beat_frames))
        if i: lo=max(lo,(anchors[i-1]+anchor)//2+1)
        if i+1<len(anchors): hi=min(hi,(anchor+anchors[i+1])//2-1)
        hi=max(lo,hi)
        pre,diag=estimate(f,lo,hi,anchor,config,target_region=(ts,te),
            require_release=parsed.moras[i].kind in ('stop','affricate','nasal'))
        estimates.append((pre,diag))
    positions=interpolate_repeats([f.times[p] for p,d in estimates],parsed.moras,tempo['period_ms'])
    for i in range(len(estimates)):
        if repeated_vowel(parsed.moras,i):
            pre=int(np.argmin(abs(f.times-positions[i])))
            diag=dict(label='先行発声',selected_absolute_ms=round(float(f.times[pre]),3),
                reason='他音素の先行発声を基準に拍位置を補間',placement_method='同母音連続の拍補間',
                candidates=[dict(absolute_ms=round(float(f.times[pre]),3),score=1.,selected=True)])
            estimates[i]=(pre,diag)
    entries=[]
    for i,alias in aliases(parsed):
        anchor=anchors[i]
        ts,te,target_ok=regions[i]
        next_start=anchors[i+1] if i+1<len(anchors) else min(len(f.times),last+1)
        ceiling=max(anchor+1,next_start-1)
        ceiling=min(ceiling,len(f.times)-1)
        reasons=list(parsed.reasons)
        cap=100*parsed.confidence
        if silent: cap=0.;reasons.append('音響境界不明: 仮候補')
        if not target_ok: cap=min(cap,40.);reasons.append('後続母音が不安定')
        if i and parsed.moras[i].kind=='vowel' and parsed.moras[i-1].vowel==parsed.moras[i].vowel:
            cap=min(cap,45.);reasons.append('同母音連続: 境界が曖昧')
        if np.max(f.peak[anchors[max(0,i-1)]:ceiling+1])>=.999:
            cap=min(cap,40.);reasons.append('クリッピング候補')
        # Alignment supplies a search interval, never a forced preutterance.
        target_spectrum=np.median(f.bands[ts:te+1],axis=0)
        distance=np.linalg.norm(f.bands-target_spectrum,axis=1)
        target_match=np.exp(-3*distance)
        pre,pre_diag=estimates[i]
        lines={'preutterance':pre_diag}
        if i:
            ps,pe,previous_ok=regions[i-1]
            # Place after disappearance of the preceding consonant, with a
            # small analysis-grid safety margin inside the stable vowel.
            left_lo=min(pre,max(ps,ps+config.safety_frames))
            left_hi=min(pre,max(left_lo,pe-config.safety_frames))
            left_pref=int(np.clip(np.argmin(abs(f.times-seeds[i]["offset"])),left_lo,left_hi))
            left_quality=spectral_stable
            overlap_pref=min(pre,ps+(pe-ps)//3)
            overlap_hi=min(pre,pe)
            if not previous_ok: cap=min(cap,40.);reasons.append('前母音が不安定')
        else:
            # No previous vowel exists for '- CV'. Find quiet lead-in and the
            # onset envelope; do not borrow a medial VCV rule or fixed ms pad.
            threshold=max(f.noise_floor_db+6,float(f.power_db[anchor])-18)
            # First walk back across the rising onset envelope. The alignment
            # anchor itself can already be louder than the activity threshold.
            onset_edge=anchor
            while onset_edge>0 and f.power_db[onset_edge-1]>threshold:
                onset_edge-=1
            quiet_start=onset_edge
            while quiet_start>0 and f.power_db[quiet_start-1]<=threshold:
                quiet_start-=1
            left_lo=min(onset_edge,quiet_start+config.safety_frames) if quiet_start<onset_edge else max(0,onset_edge-config.n_fft//(2*config.hop))
            left_hi=min(pre,max(left_lo,(left_lo+anchor)//2))
            left_pref=int(np.clip(np.argmin(abs(f.times-seeds[i]["offset"])),left_lo,left_hi))
            left_quality=1-energy
            overlap_pref=min(pre,max(left_hi,(left_hi+anchor)//2))
            overlap_hi=pre
            reasons.append('先頭音: 前母音なし・無音からの立ち上がりを使用')
        off,lines['offset']=choose(f,'offset',left_lo,left_hi,left_pref,
            dict(spectral=left_quality,rms=energy if i else 1-energy,stability=spectral_stable),config,cap)
        # The requested overlap is one third of the interval from left blank
        # to preutterance. Use continuous ms below, not a coincident frame.
        if off>=pre: off=max(0,pre-1)
        lines['offset']['selected_absolute_ms']=round(float(f.times[off]),3)
        ov=off+(pre-off)/3
        lines['overlap']=dict(label='オーバーラップ',reason='左ブランクから先行発声の1/3',
            selected_absolute_ms=round(float(np.interp(ov,np.arange(len(f.times)),f.times)),3),candidates=[])
        # Fixed range begins only after the target envelope becomes stable.
        fixed_lo=max(pre,min(ts,ceiling-1))
        fixed_hi=min(ceiling-1,max(pre,ts+(te-ts)//3))
        fixed_pref=int(np.clip(np.argmin(abs(f.times-seeds[i]["consonant"])),fixed_lo,fixed_hi))
        fixed,lines['consonant']=choose(f,'consonant',fixed_lo,fixed_hi,fixed_pref,
            dict(spectral=spectral_stable*target_match,rms=energy,stability=stable),config,cap)
        # Last safe target-vowel frames precede decay or the next transition.
        end_lo=min(ceiling,max(fixed+1,ts+(te-ts)//2))
        end_hi=min(ceiling,max(end_lo,te-config.safety_frames))
        end_pref=int(np.clip(np.argmin(abs(f.times-seeds[i]["cutoff"])),end_lo,end_hi))
        end,lines['cutoff']=choose(f,'cutoff',end_lo,end_hi,end_pref,
            dict(spectral=spectral_stable*target_match,rms=energy*(1-.5*falling),stability=stable),config,cap)
        pos=dict(offset=float(f.times[off]),overlap=float(np.interp(ov,np.arange(len(f.times)),f.times)),
                 preutterance=float(f.times[pre]),consonant=float(f.times[fixed]),cutoff=float(f.times[end]))
        limit=tempo['period_ms']*config.bpm_max_shift_beats
        # Enforce the beat window for every line, including interpolated vowels.
        for key in ('offset','preutterance','consonant','cutoff'):
            pos[key]=float(np.clip(pos[key],max(0.,seeds[i][key]-limit),min(f.duration_ms,seeds[i][key]+limit)))
        pos['preutterance']=max(.003,pos['preutterance'])
        pos['offset']=min(pos['offset'],pos['preutterance']-.003)
        pos['overlap']=pos['offset']+(pos['preutterance']-pos['offset'])/3
        pos['consonant']=max(pos['preutterance'],pos['consonant'])
        for key,diag in lines.items():
            actual=round(pos[key],3)
            if actual!=diag['selected_absolute_ms']:
                diag['reason']+=' / BPM初期位置の補正幅内に制限'
                for candidate in diag['candidates']: candidate['selected']=False
                diag['candidates'].insert(0,dict(absolute_ms=actual,score=0.,selected=True,source='BPM制約'))
            diag['selected_absolute_ms']=actual
        if not lines['overlap']['candidates']:
            lines['overlap']['candidates']=[dict(absolute_ms=round(pos['overlap'],3),score=1.,selected=True)]
        if pos['cutoff']<=pos['consonant']:
            pos['cutoff']=min(f.duration_ms,pos['consonant']+.1)
            lines['cutoff']['confidence']=0.;lines['cutoff']['warning']=True
            lines['cutoff']['reason']='伸縮区間不足 / 仮候補'
            lines['cutoff']['selected_absolute_ms']=round(pos['cutoff'],3)
        for diag in lines.values():
            for key in ('confidence','warning'): diag.pop(key,None)
        for key,diag in lines.items(): reasons.append(diag['label']+': '+diag['reason'])
        offset=pos['offset']
        entries.append(dict(wav=parsed.filename,alias=alias,mora_index=i,phoneme_class=parsed.moras[i].kind,
            offset=round(offset,3),overlap=round(pos['overlap']-offset,3),
            preutterance=round(pos['preutterance']-offset,3),consonant=round(pos['consonant']-offset,3),
            cutoff=round(-(pos['cutoff']-offset),3),
            line_diagnostics=lines,reasons=reasons,initial_grid_lines_absolute_ms=seeds[i],max_correction_ms=limit,
            detected_absolute_ms=dict(offset=offset,previous_stable_start=float(f.times[regions[i-1][0]]) if i else None,
                vowel_transition=pos['preutterance'],target_stable_start=float(f.times[ts]),target_stable_end=float(f.times[te]),end=pos['cutoff']),
            evidence=dict(original_boundary_ms=float(f.times[anchor]),noise_floor_db=f.noise_floor_db),
            placement_basis='BPM初期配置→音響解析→同母音補間'))
    return entries,dict(reading=parsed.reading,moras=[asdict(m) for m in parsed.moras],
        anchors_absolute_ms=[float(f.times[a]) for a in anchors],tempo=tempo,
        coarse_anchors_absolute_ms=[float(f.times[a]) for a in coarse_anchors],
        filename_reasons=parsed.reasons)
