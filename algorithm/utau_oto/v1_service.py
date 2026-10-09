"""Non-editor workflow: analyse, merge selected estimators, then save safely."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
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
    logs=target/'ログ'
    if linked(logs):raise ValueError('リンク先のログは扱えません。')
    if logs.exists() and (not logs.is_dir() or any(linked(p) or not p.is_file() or p.suffix.lower()!='.log' for p in logs.iterdir())):
        raise ValueError('ログ保存先に管理外のファイルがあります。')
    if (target/managed_cache.MARKER).exists():
        managed_cache.clear(folder)
        return True
    # Earlier v1 builds wrote only .npz features without a management marker.
    # Never remove any directory with other files or an unknown structure.
    children=[p for p in target.iterdir() if p.name!='ログ']
    if not children:
        if not logs.exists():target.rmdir()
        return False
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
    if children[0].resolve().parent!=target.resolve():raise ValueError('キャッシュ削除先が不正です。')
    shutil.rmtree(children[0])
    if not any(target.iterdir()):target.rmdir()
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


def _alternate_folder_path(folder):
    """Return the next generated folder path without creating it."""
    parent=folder.parent
    stem='生成済み'
    for index in range(1000):
        name=stem if index==0 else f'{stem}_{folder.name}_{datetime.now():%Y%m%d_%H%M%S}_{index}'
        target=parent/name
        if not target.exists() and not target.is_symlink():return target
    raise FileExistsError('新しい保存フォルダを作れませんでした。')


def planned_output_path(folder,overwrite,existing=None):
    """Show the destination that generate() will use, without creating folders."""
    folder=Path(folder)
    if existing is None:existing=(folder/'oto.ini').exists()
    if existing and not overwrite:return _alternate_folder_path(folder)/'oto.ini'
    return folder/'oto.ini'


def save_oto(folder,data,initial_state,overwrite,alternate_path=None):
    folder=Path(folder)
    source=folder/'oto.ini'
    if oto_state(folder)!=initial_state:
        raise RuntimeError('解析中に既存のoto.iniが変わりました。保存せずに中止します。')
    target=source
    if target.is_symlink():raise ValueError('シンボリックリンクのoto.iniは変更できません。')
    if target.exists() and not target.is_file():raise ValueError('oto.iniが通常のファイルではありません。')
    if target.exists() and not overwrite:
        alternate=Path(alternate_path) if alternate_path is not None else _alternate_folder(folder)/'oto.ini'
        if alternate_path is not None:alternate.parent.mkdir()
        if alternate.exists() or alternate.is_symlink():
            raise FileExistsError('予定した保存先にファイルが作成されたため、oto.iniを保存できません。')
        try:atomic_write(alternate,data)
        except Exception:
            if alternate_path is not None:
                try:alternate.parent.rmdir()
                except OSError:pass
            raise
        return SaveResult(initial_state is not None,False,alternate,False,0)
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


def generate(folder,methods,overwrite,progress=None,cancel=None,device='auto',normalize_audio=True,
             planned_path=None):
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
    if progress:progress(dict(phase='WAV読込中',done=0,total=len(files)))
    entries,reports=run(folder,device=device,progress=progress,methods=methods,
                        return_data=True,cancel=cancel,
                        cache_enabled=(not output.exists() or bool(overwrite)),
                        normalize_audio=normalize_audio)
    if cancel is not None and cancel.is_set():raise InterruptedError('解析を中止しました。')
    failures={method:report['errors'] for method,report in reports.items() if report['errors']}
    if failures:
        reasons=Counter(reason for errors in failures.values() for reason in errors.values())
        summary='、'.join(f'{reason}（{count}件）' for reason,count in reasons.most_common(3))
        raise RuntimeError(f'解析できないWAVがあるため、oto.iniは保存しません。方式別失敗件数: '
            +', '.join(f'{method}={len(errors)}' for method,errors in failures.items())
            +f'。主な理由: {summary}')
    if not entries:raise RuntimeError('原音設定を生成できませんでした。')
    data=serialize(entries)
    if progress:progress(dict(phase='oto.ini生成中',done=len(files),total=len(files)))
    outcome=save_oto(folder,data,initial,overwrite,alternate_path=planned_path)
    credit_path=outcome.path.parent/'Autoto_クレジットについて.txt'
    if not credit_path.exists():
        credit=('生成した原音設定のクレジット\n\n'
                '本ソフトウェアが生成した oto.ini をそのまま公開・配布する場合は、'
                '推定に使用したソフトウェアとして「Autoto（開発：ふっきんちゃん）」の表示を必須とします。\n'
                '生成結果を編集して叩き台として使う場合の表示は任意ですが、同様に記載していただけると助かります。\n'
                '表示場所は配布ページ、同梱の説明書・クレジット一覧など、利用者が確認できる場所としてください。\n')
        try:atomic_write(credit_path,credit.encode('utf-8'))
        except (OSError,ValueError):
            if progress:progress(dict(notice='credit:oto.iniは保存しましたが、クレジット案内ファイルを保存できませんでした。READMEのクレジット条件をご確認ください。'))
    return SaveResult(outcome.source_had_oto,outcome.overwritten,outcome.path,
                      outcome.backup_created,len(entries),
                      next((str(report.get('device','cpu')) for report in reports.values()),'cpu'))
