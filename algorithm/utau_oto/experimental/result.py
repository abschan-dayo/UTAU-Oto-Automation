from dataclasses import dataclass, field, asdict

@dataclass
class PreutteranceEstimate:
    method: str
    position_ms: float | None = None
    confidence: float | None = None
    candidates: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    status: str = 'ok'

    @classmethod
    def unavailable(cls,method,reason): return cls(method,status='unavailable',metadata={'reason':str(reason)})
    def dict(self): return asdict(self)
