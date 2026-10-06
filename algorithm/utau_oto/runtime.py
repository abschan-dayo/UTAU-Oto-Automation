"""Distribution defaults, independent of the current working directory."""
import json
from pathlib import Path
import sys

def default_device(profile=None):
    import os
    override=os.environ.get('AUTOTO_DEVICE')
    if override in ('cpu','cuda','auto'):return override
    base=Path(sys._MEIPASS) if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[1]
    path=Path(profile) if profile is not None else base/'distribution.json'
    if not path.exists(): return 'auto'
    device=json.loads(path.read_text(encoding='utf-8'))['device']
    if device not in ('cpu','cuda','auto'): raise ValueError('配布設定の解析デバイスが不正です')
    return device
