"""Strict CP932 serialization. No reference oto parsing or parameter reuse."""
from pathlib import Path
import math
import os
import tempfile

FIELDS = ('offset', 'consonant', 'cutoff', 'preutterance', 'overlap')


def serialize(entries):
    lines = []
    for e in entries:
        for key in ('wav', 'alias'):
            if not e[key] or any(c in e[key] for c in '=,\r\n\x00'):
                raise ValueError(f'oto.iniに表現できない{key}: {e[key]!r}')
        values = [float(e[k]) for k in FIELDS]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('非有限のoto値です')
        offset, consonant, cutoff, pre, overlap = values
        if not (offset >= 0 and 0 <= overlap <= pre <= consonant < -cutoff):
            raise ValueError(f'oto値の位置関係が不正です: {e["alias"]}')
        numbers = [format(v if abs(v) >= .0005 else 0., '.3f').rstrip('0').rstrip('.') for v in values]
        lines.append(e['wav']+'='+e['alias']+','+','.join(numbers))
    # encode strictly: never replace an unrepresentable filename with '?'.
    return '\n'.join(lines).encode('cp932', errors='strict')


def atomic_write(path: Path, data: bytes, overwrite=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('シンボリックリンクへの出力はできません')
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} は既に存在します。--overwrite または別の出力名を指定してください')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            # Hard-link commit refuses a destination created concurrently.
            os.link(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_oto(path, data):
    """Keep exactly the previous output, without interpreting any old values."""
    path=Path(path)
    if path.name=='oto.ini':
        raise ValueError('oto.ini cannot be used as an output path; use 推定後_oto.ini instead')
    if path.is_symlink():
        raise ValueError('リンク先の原音設定には書き込めません')
    backup=path.with_name('oto_backup.ini' if path.name=='oto.ini' else path.stem+'_backup.ini')
    if path==backup:
        raise ValueError('バックアップ名を出力先には指定できません')
    if path.exists():
        if backup.is_symlink():raise ValueError('A symlink cannot be used as an output backup')
        if not backup.exists():atomic_write(backup,path.read_bytes())
    atomic_write(path,data,overwrite=True)
