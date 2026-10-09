"""Read-only wrapper: no thresholds, feature definitions or algorithm changes."""
from .result import PreutteranceEstimate
import hashlib

def wrap(entry):
    return PreutteranceEstimate('existing',float(entry['offset'])+float(entry['preutterance']),None,
        metadata={'confidence_status':'missing; beta5 does not expose a calibrated internal confidence'})

def estimate_file(path,device='auto'):
    from pathlib import Path
    from ..audio import read_wav
    from ..names import parse_name
    from ..features import Backend,extract
    from ..analysis import analyze
    path=Path(path);x,sr=read_wav(path)
    entries,report=analyze(parse_name(path.name),extract(x,sr,Backend(device)))
    from ..preutterance_calibration import calibrate_entries
    entries,report['calibration']=calibrate_entries(entries,path)
    return entries,report

class ExistingAdapter:
    """Mirror CLI's folder-level short-recording hint without writing any files."""
    def __init__(self,folder,device='auto',cache_enabled=True,normalize_audio=True):
        from pathlib import Path
        from dataclasses import replace
        import numpy as np
        from ..audio import read_wav
        from ..names import parse_name
        from ..features import Backend,extract
        from ..analysis import align
        from ..rhythm import fit_grid
        from ..config import AnalysisConfig
        self.config=AnalysisConfig();self.backend=Backend(device);self.preload={};self.cache_enabled=cache_enabled
        self.normalize_audio=normalize_audio
        files=sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower()=='.wav'),key=lambda p:p.name)
        long=[]
        for path in files:
            try:
                if len(parse_name(path.name).moras)>=6:long.append(path)
            except ValueError:pass
        periods=[]
        if long:
            for index in sorted(set(np.linspace(0,len(long)-1,min(5,len(long))).astype(int))):
                path=long[index]
                try:
                    parsed=parse_name(path.name);x,sr=read_wav(path,normalize=normalize_audio);feature=extract(x,sr,self.backend,config=self.config)
                    anchors,_,score,silent=align(feature,parsed.moras)
                    if silent:continue
                    _,tempo=fit_grid([feature.times[j] for j in anchors],parsed.moras,feature,score)
                    periods.append(tempo['period_ms']);self.preload[str(path.resolve())]=(hashlib.sha256(path.read_bytes()).hexdigest(),feature)
                except (ValueError,OSError,EOFError,RuntimeError):pass
        if len(periods)>=3:
            period=float(np.median(periods))
            if np.median(abs(np.asarray(periods)-period))/period<.1:self.config=replace(self.config,short_recording_period_ms=period)

    def estimate_file(self,path):
        from pathlib import Path
        from ..audio import read_wav
        from ..names import parse_name
        from ..features import extract
        from ..analysis import analyze
        path=Path(path);cached=self.preload.pop(str(path.resolve()),None)
        f=cached[1] if cached and cached[0]==hashlib.sha256(path.read_bytes()).hexdigest() else None
        if f is None:
            x,sr=read_wav(path,normalize=self.normalize_audio);f=extract(x,sr,self.backend,config=self.config)
        entries,report=analyze(parse_name(path.name),f,config=self.config)
        from ..preutterance_calibration import calibrate_entries
        entries,report['calibration']=calibrate_entries(entries,path,
            cache_dir=path.parent/'キャッシュ'/'音響特徴'/'existing' if self.cache_enabled else False)
        return entries,report
