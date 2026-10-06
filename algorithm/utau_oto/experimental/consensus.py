import math
import numpy as np
from .config import ConsensusConfig

def agreement(difference_ms,sigma_ms):return math.exp(-.5*(difference_ms/sigma_ms)**2)

def compare(estimates,config=None):
    c=config or ConsensusConfig()
    available={e.method:e for e in estimates if e.status=='ok' and e.position_ms is not None and math.isfinite(e.position_ms)}
    distances={};pairs=[]
    for a,b in [('existing','vision'),('existing','alignment'),('vision','alignment')]:
        distance=abs(available[a].position_ms-available[b].position_ms) if a in available and b in available else None
        distances['d_'+a+'_'+b]=distance
        if distance is not None:pairs.append(agreement(distance,c.sigma_ms))
    positions=[e.position_ms for e in available.values()]
    span=max(positions)-min(positions) if len(positions)>=2 else None
    if pairs:
        weights=c.pair_weights[:len(pairs)]
        pair_score=sum(w*s for w,s in zip(weights,sorted(pairs,reverse=True)))/sum(weights)
        span_score=agreement(span,c.sigma_ms)
        agreement_score=c.agreement_weights[0]*pair_score+c.agreement_weights[1]*span_score
    else:pair_score=span_score=agreement_score=None
    confidences=[float(e.confidence) for e in available.values() if e.confidence is not None and math.isfinite(e.confidence)]
    internal=float(np.median(confidences)) if confidences else None
    components=[(agreement_score,c.final_weights[0]),(internal,c.final_weights[1])]
    usable=[(v,w) for v,w in components if v is not None and w>0]
    final=sum(v*w for v,w in usable)/sum(w for v,w in usable) if usable else None
    consensus='3/3' if len(positions)==3 and span<=c.consensus_tolerance_ms else ('2/3' if any(d is not None and d<=c.consensus_tolerance_ms for d in distances.values()) else 'none')
    rating='UNAVAILABLE' if final is None else next((name for name,t in zip(('HIGH','GOOD','REVIEW'),c.rating_thresholds) if final>=t),'LOW')
    return dict(**distances,span_ms=span,pair_score=pair_score,span_score=span_score,agreement_score=agreement_score,
        internal_confidence=internal,final_confidence=final,consensus=consensus,rating=rating,
        available_methods=list(available),confidence_is_accuracy=False)
