from dataclasses import dataclass, field, asdict
import json,math
from pathlib import Path

@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 220
    n_mels: int = 80
    n_mfcc: int = 20
    fmin: float = 0.
    fmax: float = 8000.
    pitch_min: float = 65.
    pitch_max: float = 1100.
    top_db: float = 80.

@dataclass(frozen=True)
class VisionConfig:
    architecture: str = 'legacy'
    sequence_decoder: str = 'legacy'
    sequence_config: dict = field(default_factory=dict)
    window_ms: float = 300.
    jitter_ms: float = 100.
    gaussian_sigma_ms: float = 10.
    stride_ms: float = 75.
    peak_separation_ms: float = 25.
    activity_below_peak_db: float = 28.
    assignment_radius_beats: float = .7
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = .001
    positive_weight: float = 5.
    negatives_per_positive: float = .5
    seed: int = 42

@dataclass(frozen=True)
class AlignmentConfig:
    slot_mode: str = 'legacy'
    ordered_rhythm_mode: str = 'global'
    ordered_period_min_ms: float = 180.
    ordered_period_max_ms: float = 700.
    ordered_period_step_ms: float = 5.
    ordered_phase_step_ms: float = 10.
    ordered_tail_min_beats: float = .6
    ordered_tail_max_beats: float = 2.4
    ordered_tail_center_beats: float = 1.4
    ordered_tail_std_beats: float = .7
    ordered_novelty_window_ms: float = 50.
    ordered_tail_penalty: float = .08
    ordered_phase_penalty: float = .04
    ordered_rhythm_min_score: float = .2
    ordered_fallback_tail_max_beats: float = 3.5
    ordered_fallback_tail_center_beats: float = 1.7
    ordered_fallback_tail_std_beats: float = 1.2
    ordered_fallback_tail_penalty: float = .04
    ordered_fallback_min_gain: float = .03
    ordered_phase_mode: str = 'fixed'
    ordered_phase_power_weight: float = .5
    ordered_phase_search_ms: float = 100.
    ordered_phase_prior_weight: float = .08
    ordered_radius_beats: float = .32
    ordered_onset_power_below_peak_db: float = 18.
    ordered_onset_periodicity: float = .7
    ordered_onset_smoothing_ms: float = 25.
    ordered_boundary_prior_weight: float = 5.
    ordered_boundary_prior_std_ms: float = 70.
    ordered_phase_correction_ms: float = 0.
    vowel_min_ms: float = 15.
    vowel_max_ms: float = 900.
    consonant_min_ms: float = 5.
    consonant_max_ms: float = 220.
    consonant_mean_ms: float = 65.
    consonant_std_ms: float = 55.
    vowel_mean_ms: float = 170.
    vowel_std_ms: float = 140.
    duration_weight: float = .2
    context_before_ms: float = 220.
    context_after_ms: float = 260.
    template_pre_ms: float = 80.
    template_post_ms: float = 80.
    variance_floor: float = .15
    min_template_examples: int = 2
    gap_scale: float = .05

@dataclass(frozen=True)
class ConsensusConfig:
    sigma_ms: float = 15.
    pair_weights: tuple = (.5,.3,.2)
    agreement_weights: tuple = (.7,.3)
    final_weights: tuple = (.7,.3)
    consensus_tolerance_ms: float = 20.
    rating_thresholds: tuple = (.85,.65,.45)

@dataclass(frozen=True)
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)

    @classmethod
    def load(cls,path=None):
        data=json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls,data):
        unknown=set(data)-{'features','vision','alignment','consensus'}
        if unknown: raise ValueError('Unknown config sections: '+str(unknown))
        result=cls(**{k:t(**data.get(k,{})) for k,t in [('features',FeatureConfig),('vision',VisionConfig),('alignment',AlignmentConfig),('consensus',ConsensusConfig)]})
        f,v,a,c=result.features,result.vision,result.alignment,result.consensus
        if v.architecture not in ('legacy','frequency_context'):raise ValueError('Unknown vision architecture')
        if v.sequence_decoder not in ('legacy','ordered'):raise ValueError('Unknown vision sequence decoder')
        if v.sequence_decoder=='ordered':
            from .vision_sequence import SequenceConfig
            SequenceConfig(**v.sequence_config)
        if not (f.sample_rate==44100 and f.n_fft>=256 and f.hop_length>0 and 0<=f.fmin<f.fmax<=f.sample_rate/2 and f.n_mels>=8 and 1<=f.n_mfcc<=f.n_mels and f.top_db>0): raise ValueError('Invalid feature config')
        if not (v.window_ms>0 and 0<=v.jitter_ms<v.window_ms/2 and 0<v.stride_ms<=v.window_ms and v.gaussian_sigma_ms>0 and v.epochs>0 and v.batch_size>0): raise ValueError('Invalid vision config')
        if not (0<a.vowel_min_ms<a.vowel_max_ms and 0<a.consonant_min_ms<a.consonant_max_ms and a.variance_floor>0 and a.gap_scale>0): raise ValueError('Invalid alignment config')
        if (a.slot_mode not in ('legacy','ordered') or a.ordered_rhythm_mode not in ('global','per_wav')
                or a.ordered_phase_mode not in ('fixed','acoustic')
                or not 0<=a.ordered_phase_power_weight<=1 or a.ordered_phase_search_ms<=0
                or a.ordered_phase_prior_weight<0 or not 0<a.ordered_radius_beats<.5
                or not 0<a.ordered_onset_periodicity<=1 or a.ordered_boundary_prior_weight<0
                or a.ordered_boundary_prior_std_ms<=0 or not math.isfinite(a.ordered_phase_correction_ms)
                or not 0<a.ordered_period_min_ms<a.ordered_period_max_ms
                or a.ordered_period_step_ms<=0 or a.ordered_phase_step_ms<=0
                or not 0<a.ordered_tail_min_beats<a.ordered_tail_center_beats<a.ordered_tail_max_beats
                or a.ordered_tail_std_beats<=0 or a.ordered_novelty_window_ms<=0
                or a.ordered_tail_penalty<0 or a.ordered_phase_penalty<0
                or a.ordered_rhythm_min_score<0
                or a.ordered_fallback_tail_max_beats<=a.ordered_tail_max_beats
                or not a.ordered_tail_min_beats<a.ordered_fallback_tail_center_beats<a.ordered_fallback_tail_max_beats
                or a.ordered_fallback_tail_std_beats<=0 or a.ordered_fallback_tail_penalty<0
                or a.ordered_fallback_min_gain<0):
            raise ValueError('Invalid alignment slot config')
        if not (c.sigma_ms>0 and c.consensus_tolerance_ms>0 and len(c.pair_weights)==3 and len(c.agreement_weights)==len(c.final_weights)==2 and len(c.rating_thresholds)==3): raise ValueError('Invalid consensus config')
        for weights in (c.pair_weights,c.agreement_weights,c.final_weights):
            if min(weights)<0 or abs(sum(weights)-1)>1e-6: raise ValueError('Weights must sum to 1')
        if not 1>=c.rating_thresholds[0]>c.rating_thresholds[1]>c.rating_thresholds[2]>=0: raise ValueError('Invalid rating thresholds')
        return result

    def dict(self): return asdict(self)
