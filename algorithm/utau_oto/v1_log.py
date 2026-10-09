"""Japanese operation logs stored alongside caches, without absolute paths."""

from datetime import datetime
import os
from pathlib import Path
import re
import threading


def _linked(path):
    return path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction())


def _check_path(path):
    if any(_linked(p) for p in (path,*path.parents)):
        raise ValueError('リンク先のログ保存先は使用できません。')


def log_directory(audio_folder):
    path=Path(audio_folder).absolute()/'キャッシュ'/'ログ'
    _check_path(path)
    return path


def remove_legacy_log_directory():
    """Remove only plain .log files in the exact legacy folder, when permitted."""
    base=os.environ.get('LOCALAPPDATA')
    if not base:return False
    target=Path(base).absolute()/'UTAU原音設定ツール'/'ログ'
    try:
        _check_path(target)
        if not target.is_dir():return False
        files=list(target.iterdir())
        if any(_linked(p) or not p.is_file() or p.suffix.lower()!='.log' for p in files):return False
        if target.resolve().parent!=(Path(base).absolute()/'UTAU原音設定ツール').resolve():return False
        for path in files:path.unlink()
        target.rmdir()
        if not any(target.parent.iterdir()):target.parent.rmdir()
        return True
    except (OSError,ValueError):return False


def sanitize_log_message(message):
    absolute=re.compile(r'(?i)(?:[a-z]:[\\/]|\\\\|(?<![\w:/])/(?!/)[^\s/]+/)')
    return '\n'.join('（個人情報保護のため、パスを含む行を省略しました）'
        if absolute.search(line) else line for line in str(message).splitlines())


class OperationLog:
    def __init__(self,operation,audio_folder):
        folder=log_directory(audio_folder)
        folder.mkdir(parents=True,exist_ok=True)
        for previous in folder.glob('*.log'):
            if _linked(previous) or not previous.is_file():continue
            try:
                original=previous.read_text(encoding='utf-8')
                cleaned=sanitize_log_message(original)+('\n' if original.endswith('\n') else '')
                if cleaned!=original:previous.write_text(cleaned,encoding='utf-8')
            except (OSError,UnicodeError):pass
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in operation) or '操作'
        self.path=folder/f'{safe}_{stamp}.log'
        self.path.touch(exist_ok=False)
        self.lock=threading.Lock()
        self.write(f'操作：{operation}')
        remove_legacy_log_directory()

    def write(self,message):
        stamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines=sanitize_log_message(message).splitlines() or ['']
        with self.lock,self.path.open('a',encoding='utf-8') as stream:
            for line in lines:stream.write(f'[{stamp}] {line}\n')
