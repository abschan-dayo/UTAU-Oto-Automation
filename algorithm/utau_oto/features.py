"""5 ms grid; short-time spectrum, waveform, power and normalized ACF pitch.

FFT / autocorrelation use CUDA when available; batches bound GPU memory.
No trained alignment model or reference oto values are used.
"""
from dataclasses import dataclass
import numpy as np
import librosa
import pyworld
from scipy.ndimage import uniform_filter1d
from .config import AnalysisConfig


class Backend:
    def __init__(self, requested='auto'):
        self.device = 'cpu'
        self.reason = None
        self.torch = None
        self.cupy = None
        self.backend_name = 'NumPy'
        if requested != 'cpu':
            try:
                import cupy
                cupy.fft.rfft(cupy.zeros(1024, dtype=cupy.float64)).get()
                self.cupy = cupy
                self.device = 'cuda'
                self.backend_name = f'CuPy {cupy.__version__}'
            except Exception as cupy_error:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDAが利用できません')
                    torch.fft.rfft(torch.zeros(1024, device='cuda')).cpu()
                    self.torch = torch
                    self.device = 'cuda'
                    self.backend_name = f'PyTorch {torch.__version__}'
                except (ImportError, OSError, RuntimeError) as torch_error:
                    self.reason = f'CUDA unavailable; using CPU: {cupy_error}; {torch_error}'

    def spectrum(self, frames, n=None):
        if self.cupy is not None:
            cp = self.cupy
            return cp.asnumpy(cp.fft.rfft(cp.asarray(frames, dtype=cp.float64), n=n, axis=1))
        if self.torch is None:
            return np.fft.rfft(frames, n=n, axis=1)
        t = self.torch
        with t.inference_mode():
            return t.fft.rfft(t.as_tensor(frames, dtype=t.float64, device='cuda'), n=n, dim=1).cpu().numpy()

    def autocorrelation(self, frames):
        n = 2 * frames.shape[1]
        if self.cupy is not None:
            cp = self.cupy
            z = cp.fft.rfft(cp.asarray(frames, dtype=cp.float64), n=n, axis=1)
            return cp.asnumpy(cp.fft.irfft(z.real**2 + z.imag**2, n=n, axis=1)[:, :frames.shape[1]])
        if self.torch is None:
            z = np.fft.rfft(frames, n=n, axis=1)
            return np.fft.irfft(z.real**2 + z.imag**2, n=n, axis=1)[:, :frames.shape[1]]
        t = self.torch
        with t.inference_mode():
            z = t.fft.rfft(t.as_tensor(frames, dtype=t.float64, device='cuda'), n=n, dim=1)
            return t.fft.irfft(z.abs().square(), n=n, dim=1)[:, :frames.shape[1]].cpu().numpy()


def smooth(x, width=5):
    return uniform_filter1d(np.asarray(x,dtype=float),size=width,mode='nearest')


def scale(x):
    return np.clip(x / max(float(np.quantile(x, .95)), 1e-9), 0, 2)


@dataclass
class Features:
    times: np.ndarray
    power_db: np.ndarray
    peak: np.ndarray
    zcr: np.ndarray
    spectrum_db: np.ndarray
    bands: np.ndarray
    flux: np.ndarray
    f0: np.ndarray
    periodicity: np.ndarray
    high_ratio: np.ndarray
    duration_ms: float
    flatness: np.ndarray
    rms: np.ndarray
    noise_floor_db: float
    world_f0: np.ndarray
    frequencies_hz: np.ndarray


def extract(x, sr, backend, fmin=65., fmax=1100., config=None):
    config = config or AnalysisConfig(fmin=fmin,fmax=fmax)
    config.validate()
    fmin,fmax = config.fmin,config.fmax
    if not 30 <= fmin < fmax <= 2000:
        raise ValueError('F0範囲は 30 <= fmin < fmax <= 2000 Hz')
    hop = config.hop
    centers = np.arange(0, len(x), hop)
    short = config.n_fft
    long = max(4096, 2**int(np.ceil(np.log2(3*sr/fmin))))
    padded = np.pad(x, (long//2, long//2))
    spectrum, rms, peaks, zcr, pitches, periodicities = [], [], [], [], [], []
    lo, hi = int(sr/fmax), min(int(sr/fmin), long-2)
    lags = np.arange(lo, hi+1)
    cpu_spectrum = None
    if backend.device == 'cpu':
        cpu_spectrum = np.abs(librosa.stft(x,n_fft=short,hop_length=hop,
            window=np.hanning(short),center=True,pad_mode='constant')).T[:len(centers)]**2
    for start in range(0, len(centers), 128):
        cc = centers[start:start+128]
        frames = padded[cc[:, None] + np.arange(long)[None, :]]
        small = frames[:, long//2-short//2:long//2+short//2]
        rms.extend(np.sqrt(np.mean(small**2, axis=1)))
        peaks.extend(np.max(np.abs(small), axis=1))
        zcr.extend(np.mean(small[:, 1:]*small[:, :-1] < 0, axis=1))
        spectrum.append(cpu_spectrum[start:start+len(cc)] if cpu_spectrum is not None else
                        np.abs(backend.spectrum(small*np.hanning(short)))**2)
        centered = frames - frames.mean(axis=1, keepdims=True)
        ac = backend.autocorrelation(centered)
        energy = np.concatenate([np.zeros((len(cc), 1)), np.cumsum(centered**2, axis=1)], axis=1)
        denom = energy[:, long-lags] + energy[:, -1:] - energy[:, lags]
        nsdf = np.clip(2*ac[:, lags]/np.maximum(denom, 1e-12), -1, 1)
        for row in nsdf:
            maxima = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])) + 1
            if not len(maxima):
                pitches.append(0.); periodicities.append(0.); continue
            best = float(np.max(row[maxima]))
            # Earliest strong peak rejects multiple-period/subharmonic choices.
            candidates = maxima[row[maxima] >= max(.55, .9*best)]
            if not len(candidates):
                pitches.append(0.); periodicities.append(max(0., best)); continue
            k = int(candidates[0])
            curvature = row[k-1] - 2*row[k] + row[k+1]
            delta = .5*(row[k-1]-row[k+1])/curvature if abs(curvature) > 1e-12 else 0.
            pitches.append(float(sr/(lags[k]+np.clip(delta, -.5, .5))))
            periodicities.append(float(row[k]))
    spec = np.concatenate(spectrum)
    power = 20*np.log10(np.maximum(rms, 1e-10))
    f0 = np.asarray(pitches)
    periodicity = np.asarray(periodicities)
    quiet = power < max(-65, float(np.max(power))-45)
    f0[quiet] = 0.; periodicity[quiet] = 0.
    freq = np.fft.rfftfreq(short, 1/sr)
    edges = [0, 200, 400, 700, 1100, 1600, 2300, 3200, 4500, 6500, 10000, 22051]
    bands = np.stack([spec[:, (freq >= a) & (freq < b)].sum(axis=1) for a,b in zip(edges[:-1],edges[1:])], axis=1)
    # Normalized envelopes retain formant changes independent of loudness.
    bands = np.sqrt(bands / np.maximum(bands.sum(axis=1, keepdims=True), 1e-12))
    padded_bands = np.pad(bands, ((3,3),(0,0)), mode='edge')
    flux = np.linalg.norm(padded_bands[6:] - padded_bands[:-6], axis=1)
    high = spec[:, freq >= 3000].sum(axis=1)/np.maximum(spec.sum(axis=1), 1e-12)
    world_pitch,world_times = pyworld.harvest(np.ascontiguousarray(x,dtype=np.float64),sr,
        f0_floor=fmin,f0_ceil=fmax,frame_period=hop*1000/sr)
    world_pitch = pyworld.stonemask(np.ascontiguousarray(x,dtype=np.float64),world_pitch,world_times,sr)
    nearest = np.searchsorted((world_times[:-1]+world_times[1:])/2,centers/sr)
    world_pitch = world_pitch[nearest]
    flatness = librosa.feature.spectral_flatness(S=np.sqrt(spec).T,power=2)[0]
    rms_values = librosa.feature.rms(y=x,frame_length=short,hop_length=hop,
        center=True,pad_mode='constant')[0][:len(centers)]
    # Low percentile is a local noise estimate, not an absolute calibration.
    floor = max(-100.,min(float(np.quantile(power,.15)),float(np.max(power))-25.))
    return Features(centers*1000/sr, power, np.asarray(peaks), np.asarray(zcr),
                    10*np.log10(np.maximum(spec, 1e-12)), bands, smooth(flux),
                    f0, periodicity, high, len(x)*1000/sr,flatness,rms_values,floor,world_pitch,freq)
