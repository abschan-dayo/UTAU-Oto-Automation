"""Deterministic settings, independent of future preset/UI layers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    hop: int = 220
    n_fft: int = 1024
    fmin: float = 65.
    fmax: float = 1100.
    stable_frames: int = 4
    safety_frames: int = 2
    candidate_distance: int = 4
    bpm_max_shift_beats: float = .15
    short_recording_period_ms: float | None = None
    overall_review_percent: float = 60.
    line_warning_percent: float = 50.

    def validate(self):
        if not 30 <= self.fmin < self.fmax <= 2000:
            raise ValueError('F0範囲は 30 <= fmin < fmax <= 2000 Hz')
        if not 0 <= self.bpm_max_shift_beats <= .3:
            raise ValueError('BPM補正幅は0〜0.3拍の範囲')
        if self.hop < 1 or self.n_fft < 256 or self.stable_frames < 1 or self.safety_frames < 0:
            raise ValueError('解析設定の値が不正です')
