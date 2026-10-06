"""Human-readable Japanese operation logs outside the source voicebank."""

from datetime import datetime
import os
from pathlib import Path
import threading


def log_directory():
    base=os.environ.get('LOCALAPPDATA')
    if base:
        return Path(base)/'UTAU原音設定ツール'/'ログ'
    return Path.home()/'.utau-auto-oto'/'ログ'


class OperationLog:
    def __init__(self,operation):
        folder=log_directory()
        folder.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path=folder/f'{stamp}_{os.getpid()}.log'
        self.lock=threading.Lock()
        self.write(f'操作：{operation}')

    def write(self,message):
        stamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines=str(message).splitlines() or ['']
        with self.lock,self.path.open('a',encoding='utf-8') as stream:
            for line in lines:stream.write(f'[{stamp}] {line}\n')
