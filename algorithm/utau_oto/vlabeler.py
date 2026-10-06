"""OpenUtau-compatible vLabeler IPC bridge (ZeroMQ request/reply)."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from .writer import atomic_write


def settings_path():
    if sys.platform=='darwin':
        base=Path.home()/'Library'/'Application Support'
    else:
        base=Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')
    return base/'UTAU原音設定自動化ツール'/'設定.json'


def _valid_executable(path):
    if path is None:return False
    if sys.platform=='darwin':return path.name.lower()=='vlabeler.app' and path.is_dir()
    return path.name.lower()=='vlabeler.exe' and path.is_file()


def saved_executable():
    path=settings_path()
    try:
        value=json.loads(path.read_text(encoding='utf-8')).get('vlabeler_exe')
        candidate=Path(value) if value else None
        return candidate if _valid_executable(candidate) else None
    except (OSError,ValueError,TypeError):return None


def save_executable(executable):
    executable=Path(executable).resolve()
    if not _valid_executable(executable):
        raise ValueError('vLabeler.appを指定してください。' if sys.platform=='darwin' else 'vLabeler.exeを指定してください。')
    path=settings_path()
    atomic_write(path,json.dumps({'vlabeler_exe':str(executable)},ensure_ascii=False).encode('utf-8'),overwrite=True)


def existing_project(folder):
    projects=[p for p in Path(folder).glob('*.lbp') if p.is_file() and not p.is_symlink()]
    return max(projects,key=lambda p:p.stat().st_mtime_ns) if projects else None


def installed_labeler_name(executable,alternate):
    """Use the identifier shipped with this vLabeler, which can vary by version."""
    folder='oto-labeler' if alternate else 'utau-singer-labeler'
    executable=Path(executable)
    root=executable.parent
    candidates=[root/'app'/'resources'/'labelers'/folder/'labeler.json',
                root/'resources'/'labelers'/folder/'labeler.json']
    if executable.suffix.lower()=='.app':
        candidates[:0]=[executable/'Contents'/'app'/'resources'/'labelers'/folder/'labeler.json',
                        executable/'Contents'/'Resources'/'app'/'resources'/'labelers'/folder/'labeler.json']
    for path in candidates:
        if path.is_file():
            try:
                name=json.loads(path.read_text(encoding='utf-8'))['name']
            except (OSError,ValueError,KeyError) as error:
                raise RuntimeError(f'vLabelerのラベラー定義を読み取れません: {path}') from error
            if not isinstance(name,str) or not name:
                raise RuntimeError(f'vLabelerのラベラー名が不正です: {path}')
            return name
    raise RuntimeError(f'vLabelerの{folder}定義が見つかりません。vLabelerの場所を確認してください。')


def request_message(project_file,sample_folder,oto_file,labeler_name=None):
    """Use vLabeler's published OpenOrCreate structure; never create .lbp ourselves."""
    source=Path(sample_folder)
    project=Path(project_file)
    alternate=Path(oto_file).parent!=source
    args=dict(labelerName=labeler_name or ('oto-plus.default' if alternate else 'utau-singer.default'),
              sampleDirectory=str(source),cacheDirectory=str(source/'キャッシュ'/'vLabeler') if not alternate else None,
              encoding='Shift_JIS',autoExport=False)
    if alternate:args['inputFile']=str(oto_file)
    else:args['labelerParams']={'useRootDirectory':{'type':'boolean','value':True}}
    return dict(type='OpenOrCreate',projectFile=str(project),newProjectArgs=args,
                sentAt=int(time.time()*1000))


def _send(request,timeout_ms=1000):
    import zmq
    context=zmq.Context.instance()
    socket=context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER,0)
    socket.setsockopt(zmq.SNDTIMEO,timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO,timeout_ms)
    try:
        socket.connect('tcp://127.0.0.1:32342')
        socket.send_string(json.dumps(request,ensure_ascii=False))
        return socket.recv_string()
    finally:socket.close()


def open_in_vlabeler(executable,source_folder,oto_file):
    executable=Path(executable)
    if not _valid_executable(executable):raise FileNotFoundError('vLabelerが見つかりません。')
    source=Path(source_folder)
    output=Path(oto_file)
    old=existing_project(source) if output.parent==source else None
    labeler_name=installed_labeler_name(executable,output.parent!=source) if old is None else None
    try:_send(dict(type='Heartbeat',sentAt=int(time.time()*1000)))
    except Exception:
        command=['open','-a',str(executable)] if sys.platform=='darwin' else [str(executable)]
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        subprocess.Popen(command,cwd=executable.parent,creationflags=flags,
                         stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(50):
            time.sleep(.2)
            try:
                _send(dict(type='Heartbeat',sentAt=int(time.time()*1000)))
                break
            except Exception:continue
        else:raise RuntimeError('vLabelerの起動を確認できませんでした。')
    project=old or (output.parent/'_vlabeler.lbp')
    reply=_send(request_message(project,source,output,labeler_name))
    return project,reply
