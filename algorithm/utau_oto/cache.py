"""Owned, replace-as-a-whole cache. No source audio or persistent user data."""
from pathlib import Path
import json
import os
import shutil
import tempfile

MARKER='管理情報.json'
MAGIC='utau-auto-oto-regenerable-cache-v1'


def _remove_owned(path,parent):
    path=Path(path)
    if path.resolve().parent!=Path(parent).resolve() or _linked(path):
        raise ValueError('キャッシュ削除先が想定フォルダ外です')
    shutil.rmtree(path)


def _linked(path):
    return path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction())


def validate(folder):
    target=Path(folder).absolute()/'キャッシュ'
    if not target.exists() and not _linked(target): return target
    if _linked(target) or not target.is_dir():
        raise ValueError('キャッシュが通常のフォルダではありません')
    marker=target/MARKER
    logs_only=not marker.exists() and all(p.name=='ログ' for p in target.iterdir())
    if not logs_only and (not marker.is_file() or _linked(marker)):
        raise ValueError('既存の「キャッシュ」は本ツール管理外です。別の場所へ移してから再実行してください')
    info={'kind':MAGIC,'files':[]} if logs_only else json.loads(marker.read_text(encoding='utf-8'))
    if info.get('kind')!=MAGIC or not isinstance(info.get('files'),list):
        raise ValueError('キャッシュ管理情報が不正です')
    actual=[]
    for base,dirs,files in os.walk(target,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name
            if _linked(p): raise ValueError('キャッシュ内にリンクがあるため更新・削除を中止しました')
        for name in files:
            p=Path(base)/name
            if p.suffix.lower() in ('.wav','.ini'):
                raise ValueError('キャッシュ内に保護対象ファイルがあります')
            relative=p.relative_to(target).parts
            if relative[0]=='ログ':
                if len(relative)!=2 or p.suffix.lower()!='.log':
                    raise ValueError('ログ保存先に管理外ファイルがあります')
                continue
            if p!=marker: actual.append(p.relative_to(target).as_posix())
    # Missing generated files are harmless; newly added files are never erased.
    if not set(actual).issubset(set(info['files'])):
        raise ValueError('キャッシュ内に管理外ファイルがあります。移動してから再実行してください')
    return target


def clear(folder):
    target=validate(folder)
    if target.exists():
        for child in target.iterdir():
            if child.name=='ログ':continue
            if child.is_dir():_remove_owned(child,target)
            else:child.unlink()
        if not any(target.iterdir()):target.rmdir()


class CacheBuild:
    def __init__(self,folder):
        self.target=validate(folder)
        self.stage=Path(tempfile.mkdtemp(prefix='.auto_oto-cache-',dir=self.target.parent))
        self.published=False

    def finish(self,metadata):
        files=sorted(p.relative_to(self.stage).as_posix() for p in self.stage.rglob('*') if p.is_file())
        info=dict(kind=MAGIC,files=files,**metadata)
        (self.stage/MARKER).write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def publish(self):
        # Validate again in case the user added a file during analysis.
        validate(self.target.parent)
        logs=self.target/'ログ'
        if logs.is_dir():
            shutil.copytree(logs,self.stage/'ログ',dirs_exist_ok=True)
        old=None
        if self.target.exists():
            old=Path(tempfile.mkdtemp(prefix='.auto_oto-old-cache-',dir=self.target.parent))
            old.rmdir()
            self.target.rename(old)
        try:
            self.stage.rename(self.target)
            self.published=True
        except OSError:
            if old: old.rename(self.target)
            raise
        if old:
            # old is an exact rename of the just-validated, owned cache.
            _remove_owned(old,self.target.parent)

    def close(self):
        if not self.published and self.stage.exists():
            # Only the exact staging directory created by this instance.
            _remove_owned(self.stage,self.target.parent)
