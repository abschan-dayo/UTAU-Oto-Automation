"""Read-only WAV normalization for analysis; source audio is never modified."""
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_RATE = 44100


def read_wav(path: Path, normalize=True):
    """Read mono float audio, optionally normalizing to 44.1 kHz / PCM-16 precision.

    Channel downmixing is always in memory because estimators require mono audio.
    Normalization is in memory only; the original WAV remains byte-for-byte intact.
    """
    try:
        with sf.SoundFile(str(path), 'r') as w:
            if w.format not in ('WAV', 'WAVEX'):
                raise ValueError('WAV形式の音声ファイルを指定してください')
            source_rate, channels, subtype = w.samplerate, w.channels, w.subtype
            expected = len(w)
            data = w.read(dtype='float64', always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError(f'WAVを読み込めません: {e}') from e
    if len(data) != expected or len(data) < 441 or not np.isfinite(data).all():
        raise ValueError('WAVが短すぎるか、データが不正です')

    if channels != 1:
        # Average channels for predictable mono conversion, including multichannel files.
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if not normalize:
        return data, source_rate
    if source_rate != TARGET_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        factor = gcd(source_rate, TARGET_RATE)
        data = resample_poly(data, TARGET_RATE // factor, source_rate // factor)

    # Match signed 16-bit PCM quantization precision while keeping float input for DSP.
    data = np.clip(data, -1.0, 1.0 - 1.0 / 32768.0)
    data = (data * 32768.0).round() / 32768.0
    if len(data) < 441 or not np.isfinite(data).all():
        raise ValueError('変換後のWAVが短すぎるか、データが不正です')
    return data, TARGET_RATE


def normalization_notice(path: Path):
    """Describe required conversion without running the signal resampler."""
    try:
        info = sf.info(str(path))
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError(f'WAVを読み込めません: {e}') from e
    if info.format not in ('WAV', 'WAVEX'):
        raise ValueError('WAV形式の音声ファイルを指定してください')
    changed = []
    if info.channels != 1:
        changed.append(f'{info.channels}ch→Mono')
    if info.samplerate != TARGET_RATE:
        changed.append(f'{info.samplerate}Hz→44100Hz')
    if info.subtype != 'PCM_16':
        changed.append(f'{info.subtype}→16bit相当')
    return ', '.join(changed) if changed else None


def source_format(path: Path):
    """Return a compact format description without exposing the source path."""
    info=sf.info(str(path))
    return f'{info.samplerate} Hz / {info.subtype} / {info.channels} ch'
