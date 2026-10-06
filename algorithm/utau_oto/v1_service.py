"""Non-editor workflow: analyse, merge selected estimators, then save safely."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil

from .writer import atomic_write,serialize
from .experimental.parallel import run
from . import cache as managed_cache


@dataclass(frozen=True)
class SaveResult:
    source_had_oto: bool
    overwritten: bool
    path: Path
    backup_created: bool
    entries: int
    device: str = 'cpu'


def check_folder(folder):
    if Path(folder).is_symlink():raise ValueError('リンク先のフォルダは使用できません。')
    folder=Path(folder).resolve()
    if not folder.is_dir():raise ValueError('WAVが入った音階フォルダを指定してください。')
    if folder.is_symlink():raise ValueError('リンク先のフォルダは使用できません。')
    files=sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.wav'),key=lambda p:p.name)
    if not files:raise ValueError('フォルダ直下にWAVがありません。')
    return folder,files


def oto_state(folder):
    path=Path(folder)/'oto.ini'
    if path.is_symlink():raise ValueError('リンク先のoto.iniは変更できません。')
    if path.exists() and not path.is_file():raise ValueError('oto.iniが通常のファイルではありません。')
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def clear_analysis_cache(folder):
    """Remove only the tool's cache; refuse unfamiliar or linked content."""
    folder=Path(folder).resolve()
    target=folder/'キャッシュ'
    linked=lambda path: path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction())
    if linked(target):raise ValueError('リンク先のキャッシュは削除できません。')
    if not target.exists():return False
    if not target.is_dir():raise ValueError('キャッシュが通常のフォルダではありません。')
    if (target/managed_cache.MARKER).exists():
        managed_cache.clear(folder)
        return True
    # Earlier v1 builds wrote only .npz features without a management marker.
    # Never remove any directory with other files or an unknown structure.
    children=list(target.iterdir())
    if not children:
        target.rmdir();return True
    if len(children)!=1 or children[0].name!='音響特徴' or linked(children[0]) or not children[0].is_dir():
        raise ValueError('管理外のファイルがあるため、キャッシュを削除しませんでした。')
    for path in children[0].rglob('*'):
        if linked(path):raise ValueError('リンクを含むため、キャッシュを削除しませんでした。')
        parts=path.relative_to(children[0]).parts
        if path.is_dir():
            if len(parts)!=1 or parts[0] not in ('existing','vision','alignment'):
                raise ValueError('管理外のフォルダがあるため、キャッシュを削除しませんでした。')
        elif not path.is_file() or path.suffix.lower()!='.npz' or len(parts)>2:
            raise ValueError('管理外のファイルがあるため、キャッシュを削除しませんでした。')
    shutil.rmtree(target)
    return True


def _alternate_folder(folder):
    """Never change an existing generated folder or its possible user edits."""
    parent=folder.parent
    stem='生成済み'
    for index in range(1000):
        name=stem if index==0 else f'{stem}_{folder.name}_{datetime.now():%Y%m%d_%H%M%S}_{index}'
        target=parent/name
        try:target.mkdir()
        except FileExistsError:continue
        return target
    raise FileExistsError('新しい保存フォルダを作れませんでした。')


def save_oto(folder,data,initial_state,overwrite):
    folder=Path(folder)
    source=folder/'oto.ini'
    if oto_state(folder)!=initial_state:
        raise RuntimeError('解析中に既存のoto.iniが変わりました。保存せずに中止します。')
    target=source
    if target.is_symlink():raise ValueError('シンボリックリンクのoto.iniは変更できません。')
    if target.exists() and not target.is_file():raise ValueError('oto.iniが通常のファイルではありません。')
    if target.exists() and not overwrite:
        raise FileExistsError('既存のoto.iniを上書きするかどうか選んでください。')
    backup_created=False
    if target.exists():
        backup=folder/'oto_backup.ini'
        if not backup.exists() and not backup.is_symlink():
            atomic_write(backup,target.read_bytes())
            backup_created=True
        atomic_write(target,data,overwrite=True)
    else:
        atomic_write(target,data)
    return SaveResult(initial_state is not None,target.exists() and overwrite,target,backup_created,0)


def generate(folder,methods,overwrite,progress=None,cancel=None,device='auto'):
    folder,files=check_folder(folder)
    methods=tuple(dict.fromkeys(methods))
    if not methods or not set(methods)<={'existing','vision','alignment'}:
        raise ValueError('解析方式を1つ以上選択してください。')
    initial=oto_state(folder)
    output=folder/'oto.ini'
    if output.is_symlink():raise ValueError('シンボリックリンクのoto.iniは変更できません。')
    if output.exists() and not output.is_file():raise ValueError('oto.iniが通常のファイルではありません。')
    if output.exists() and overwrite is None:
        raise ValueError('既存のoto.iniの上書き可否を指定してください。')
    if output.exists() and not overwrite:
        raise FileExistsError('oto.iniを保持するため、解析を中止しました。')
    if progress:progress(dict(phase='WAV読込中',done=0,total=len(files)))
    entries,reports=run(folder,device=device,progress=progress,methods=methods,
                        return_data=True,cancel=cancel,
                        cache_enabled=(not output.exists() or bool(overwrite)))
    if cancel is not None and cancel.is_set():raise InterruptedError('解析を中止しました。')
    failures={method:report['errors'] for method,report in reports.items() if report['errors']}
    if failures:
        raise RuntimeError('解析できないWAVがあったため、oto.iniは保存しません。詳細: '+json.dumps(failures,ensure_ascii=False))
    if not entries:raise RuntimeError('原音設定を生成できませんでした。')
    data=serialize(entries)
    if progress:progress(dict(phase='oto.ini生成中',done=len(files),total=len(files)))
    outcome=save_oto(folder,data,initial,overwrite)
    return SaveResult(outcome.source_had_oto,outcome.overwritten,outcome.path,
                      outcome.backup_created,len(entries),
                      next((str(report.get('device','cpu')) for report in reports.values()),'cpu'))
