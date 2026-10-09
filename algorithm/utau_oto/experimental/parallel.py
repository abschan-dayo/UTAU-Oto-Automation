"""Three estimator threads; only the coordinator writes outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import threading,json,argparse,os,statistics
import time
from datetime import datetime
from collections import Counter
from ..writer import serialize,write_oto,atomic_write

def worker(method,folder,device,model,cache,progress=None,cancel=None,normalize_audio=True):
    started=time.monotonic()
    def emit(**event):
        if progress:progress(dict(method=method,**event))
    files=sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower()=='.wav'),key=lambda p:p.name)
    emit(done=0,total=len(files),message='準備中')
    from ..names import parse_name,aliases
    if method=='existing':
        from .existing_adapter import ExistingAdapter
        estimator=ExistingAdapter(folder,device,cache_enabled=cache is not None,normalize_audio=normalize_audio)
        actual_device=estimator.backend.device
        backend_name=estimator.backend.backend_name
    else:
        from .features import extract_features
        if method=='vision':
            from .vision_estimator import VisionEstimator
            estimator=VisionEstimator(model,device)
            actual_device=estimator.device if not estimator.error else 'unavailable'
            backend_name='PyTorch' if not estimator.error else 'unavailable'
        else:
            from .alignment import AlignmentEstimator
            estimator=AlignmentEstimator(model)
            actual_device='cpu'
            backend_name='NumPy'
    results={};errors={}
    for index,path in enumerate(files):
        if cancel is not None and cancel.is_set():break
        emit(done=index,total=len(files),message=path.name,step='WAV読込中')
        try:
            if method=='existing':
                emit(done=index,total=len(files),message=path.name,step='境界解析中')
                results[path.name]=estimator.estimate_file(path)[0]
            else:
                if estimator.error:raise ValueError(estimator.error)
                emit(done=index,total=len(files),message=path.name,step='特徴抽出中')
                features=extract_features(path,estimator.config.features,Path(cache)/method if cache else None,
                                          normalize_audio=normalize_audio)
                emit(done=index,total=len(files),message=path.name,step='境界解析中')
                seen=Counter();rows=[]
                for _,alias in aliases(parse_name(path.name)):
                    result=estimator.estimate(features,path.name,alias,seen[alias]);seen[alias]+=1
                    rows.append(dict(alias=alias,estimate=result.dict()))
                results[path.name]=rows
        except Exception as e:errors[path.name]=str(e)
        emit(done=index+1,total=len(files),message=path.name,errors=len(errors))
    return dict(results=results,errors=errors,pid=os.getpid(),thread_id=threading.get_ident(),
                device=actual_device,backend=backend_name,elapsed_s=round(time.monotonic()-started,2),
                processed=len(results)+len(errors))


def error_category(message):
    value=str(message).lower()
    if '標準外wav' in value or 'pcm' in value or 'sample rate' in value:
        return '入力形式'
    if 'wav' in value or 'audio' in value or 'soundfile' in value:
        return '音声の読み込み・解析'
    if 'model' in value or 'template' in value or 'checkpoint' in value:
        return 'モデル・テンプレート'
    return 'その他'

def variant(entries,predictions):
    """Preserve other parameters; reject unsafe replacements explicitly."""
    output=[];notes=[];seen=Counter();lookup={}
    for row in predictions:
        alias=row['alias'];key=(alias,seen[alias]);seen[alias]+=1;lookup[key]=row['estimate']
    seen.clear()
    for entry in entries:
        e=dict(entry);key=(e['alias'],seen[e['alias']]);seen[e['alias']]+=1
        estimate=lookup.get(key);reason=None
        if not estimate or estimate['status']!='ok':reason='推定できないため①を使用'
        else:
            pre=estimate['position_ms']-e['offset']
            if e['overlap']<=pre<=e['consonant']:e['preutterance']=pre
            else:reason='他の4線との位置関係が不正になるため①を使用'
        notes.append(dict(alias=e['alias'],estimate=estimate,fallback=reason))
        output.append(e)
    return output,notes


def output_name(method,selected):
    """Use a distinct name for every generated estimator result."""
    if len(selected)==1:return 'oto.ini'
    return {'existing':'oto_①既存.ini','vision':'oto_②Vision.ini',
            'alignment':'oto_③Alignment.ini'}[method]


def median_variant(method_entries,selected=None):
    """Median of selected estimates; statistics.median averages two values."""
    selected=tuple(selected or method_entries)
    baseline=method_entries['existing']
    result=[]
    for index,entry in enumerate(baseline):
        row=dict(entry)
        row['preutterance']=statistics.median(method_entries[method][index]['preutterance']
                                              for method in selected)
        result.append(row)
    return result


def new_result_folder(folder):
    """Keep every run distinct when several pitches share one parent directory."""
    name='生成済み_'+folder.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    for number in range(1000):
        candidate=folder.parent/(name+(f'_{number+1}' if number else ''))
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError('結果フォルダの名前を確保できませんでした')

def run(folder,device='auto',vision=None,alignment=None,progress=None,methods=None,
        return_data=False,cancel=None,cache_enabled=True,normalize_audio=True):
    methods=tuple(methods or ('existing','vision','alignment'))
    if not methods or not set(methods)<= {'existing','vision','alignment'}:raise ValueError('Unknown method')
    def emit(**event):
        if progress:progress(event)
    emit(phase='WAV読込中')
    folder=Path(folder).resolve()
    if not folder.is_dir():raise ValueError('原音フォルダを指定してください')
    if not any(p.suffix.lower()=='.wav' for p in folder.iterdir() if p.is_file()):raise ValueError('WAVがありません')
    app=Path(__file__).resolve().parents[2]
    models=app/'models'
    all_paths={'existing':None,'vision':str(vision or models/'vision.pt'),'alignment':str(alignment or models/'alignment.npz')}
    # Method 1 supplies the other four lines even when only 2 or 3 is selected.
    paths={m:all_paths[m] for m in ('existing','vision','alignment') if m=='existing' or m in methods}
    collected={}
    # librosa/numba can inspect torch during their own lazy imports. Complete
    # optional torch initialization on the coordinator before workers race it.
    try:
        import torch
    except ImportError:
        pass
    from .features import extract_features
    from .config import Config
    from .. import features as core_features, analysis as core_analysis
    # Initialize the CUDA runtime before estimator threads start. CuPy's first
    # FFT and kernel compilation are not reliable when initialized concurrently.
    backend = core_features.Backend(device)
    emit(diagnostic='backend',device=backend.device,backend=backend.backend_name,
         gpu=backend.gpu_name or '使用なし',fallback_reason=backend.reason)
    if backend.reason:
        emit(notice='device:'+backend.reason)
        print(backend.reason, flush=True)
    device = backend.device
    # Exercise lazy numerical imports once; an existing disk cache would skip
    # that initialization, so deliberately omit cache for this warm-up.
    for wav in sorted(folder.iterdir()):
        if wav.is_file() and wav.suffix.lower()=='.wav':
            try:
                extract_features(wav,Config().features,None,normalize_audio=normalize_audio)
                break
            except (ValueError,OSError,RuntimeError):
                continue
    print('選択した方式を解析しています：'+', '.join(methods),flush=True)
    from ..audio import normalization_notice,source_format
    changes=[]
    formats=Counter()
    unreadable=0
    for wav in sorted(folder.iterdir()):
        if wav.is_file() and wav.suffix.lower()=='.wav':
            try:
                formats[source_format(wav)]+=1
                change=normalization_notice(wav) if normalize_audio else None
                if change:changes.append(f'{wav.name}: {change}')
            except (ValueError,OSError,RuntimeError):
                unreadable+=1
    emit(diagnostic='audio',formats=dict(formats),converted=len(changes),unreadable=unreadable,
         normalize_audio=normalize_audio)
    if changes:
        notice='入力WAVを解析用に44.1 kHz・16 bit相当・Monoへ変換します（元ファイルは変更しません）:\n'+'\n'.join(changes[:20])
        if len(changes)>20:notice+=f'\nほか {len(changes)-20} 件'
        emit(notice='audio:'+notice)
        print(notice.replace('\n',' / '),flush=True)
    with ThreadPoolExecutor(max_workers=3,thread_name_prefix='oto-estimator') as pool:
        emit(phase='境界解析中')
        cache=str(folder/'キャッシュ'/'音響特徴') if cache_enabled else None
        futures={pool.submit(worker,m,str(folder),device,paths[m],cache,progress,cancel,normalize_audio):m for m in paths}
        for future in as_completed(futures):
            method=futures[future]
            try:collected[method]=future.result()
            except Exception as e:collected[method]=dict(results={},errors={'worker':str(e)})
            report=collected[method]
            emit(diagnostic='worker',method=method,success=len(report['results']),
                 failed=len(report['errors']),skipped=0,
                 error_reasons=dict(Counter(error_category(e) for e in report['errors'].values())),
                 elapsed_s=report.get('elapsed_s'),
                 device=report.get('device','unknown'),backend=report.get('backend','unknown'))
            print(method+' 完了',flush=True)
    if cancel is not None and cancel.is_set():raise InterruptedError('解析を中止しました')
    base=collected['existing']['results']
    if not base:
        reasons=Counter(collected['existing']['errors'].values())
        summary='、'.join(f'{reason}（{count}件）' for reason,count in reasons.most_common(3))
        raise RuntimeError(f'音響解析に成功したWAVはありません（失敗{len(collected["existing"]["errors"])}件）。{summary}')
    emit(phase='統合中')
    reports={};payloads={};method_entries={}
    for method in paths:
        entries=[];notes={}
        for filename,rows in base.items():
            if method=='existing':
                entries.extend(dict(e) for e in rows)
            else:
                converted,notes[filename]=variant(rows,collected[method]['results'].get(filename,[]));entries.extend(converted)
        method_entries[method]=entries
        if method in methods:payloads[output_name(method,methods)]=serialize(entries)
        fallback=Counter(row['fallback'] for rows in notes.values() for row in rows if row['fallback'])
        reports[method]=dict(errors=collected[method]['errors'],pid=collected[method].get('pid'),thread_id=collected[method].get('thread_id'),execution='threads',entries=len(entries),details=notes,
            device=collected[method].get('device','unknown'),backend=collected[method].get('backend','unknown'),
            elapsed_s=collected[method].get('elapsed_s'),success_wavs=len(collected[method]['results']),
            failed_wavs=len(collected[method]['errors']),adopted=len(entries)-sum(fallback.values()) if method!='existing' else len(entries),
            fallback_reasons=dict(fallback))
        emit(diagnostic='method',method=method,adopted=reports[method]['adopted'],
             fallback_reasons=dict(fallback),failed_wavs=reports[method]['failed_wavs'])
    merged=median_variant(method_entries,methods)
    if return_data:
        return merged,reports
    payloads['oto_中央値.ini']=serialize(merged)
    output=new_result_folder(folder)
    for name,data in payloads.items():write_oto(output/name,data)
    atomic_write(output/'処理結果.json',json.dumps(reports,ensure_ascii=False,indent=2).encode('utf-8'),overwrite=True)
    fallback=sum(bool(r['fallback']) for method in ('vision','alignment') if method in reports for rows in reports[method]['details'].values() for r in rows)
    print(f'保存先: {output}\n②③で基準値を代用した設定: {fallback}件（詳細は処理結果.json）',flush=True)
    emit(phase=f'保存完了：{output}\n②③で基準値を代用した設定：{fallback}件')
    return reports

def main():
    p=argparse.ArgumentParser(description='音階フォルダの親に結果フォルダを作り、選択方式と3方式の中央値を保存')
    from ..runtime import default_device
    p.add_argument('folder',type=Path);p.add_argument('--device',choices=['cpu','cuda','auto'],default=default_device())
    p.add_argument('--vision',type=Path);p.add_argument('--alignment',type=Path)
    p.add_argument('--no-progress',action='store_true',help='進捗ウィンドウを表示しない')
    p.add_argument('--methods',nargs='+',choices=['1','2','3'])
    a=p.parse_args()
    mapping={'1':'existing','2':'vision','3':'alignment'}
    methods=[mapping[m] for m in a.methods] if a.methods else None
    if not a.no_progress:
        from .progress import launch,choose_methods
        if methods is None:
            methods=choose_methods()
            if not methods:return 0
        return launch(lambda callback:run(a.folder,a.device,a.vision,a.alignment,callback,methods),methods=('existing','vision','alignment'))
    try:run(a.folder,a.device,a.vision,a.alignment,methods=methods)
    except Exception as e:print('エラー: '+str(e),flush=True);return 1
    return 0
