"""Read-only libsndfile ingestion. Never convert or rename source audio."""
from pathlib import Path
import soundfile as sf
import numpy as np


def read_wav(path: Path):
    try:
        with sf.SoundFile(str(path), 'r') as w:
            if (w.samplerate, w.channels, w.subtype) != (44100, 1, 'PCM_16') or w.format not in ('WAV','WAVEX'):
                raise ValueError('標準外WAV: 44.1kHz / 16bit / mono PCM が必要です。元WAVは変更していません')
            expected = len(w)
            x = w.read(dtype='float64', always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as e:
        raise ValueError(f'WAVを読めません: {e}') from e
    if len(x) != expected or len(x) < 441 or not np.isfinite(x).all():
        raise ValueError('WAVが短すぎるか、データが不正です')
    return x, 44100
