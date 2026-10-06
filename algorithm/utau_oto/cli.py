import argparse
from dataclasses import asdict,replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
from . import __version__
from .audio import read_wav
from .audio import normalization_notice
from .names import parse_name
from .features import Backend,extract
from .analysis import analyze
from .config import AnalysisConfig
from .writer import serialize,write_oto
from .cache import CacheBuild,clear
from .runtime import default_device


def main(argv=None):
    p=argparse.ArgumentParser(description='UTAU原音設定 コアβ（フォルダ直下・再帰なし）')
    p.add_argument('folder',type=Path)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default=default_device())
    p.add_argument('--output',type=Path,help='既定: 入力フォルダ内のoto.ini')
    p.add_argument('--overwrite',action='store_true',help='Allow replacing the existing oto.ini')
    p.add_argument('--save-features',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--name-map',type=Path,help='UTF-8 JSON: WAV名と録音かな列の対応')
    p.add_argument('--fmin',type=float,default=65.)
    p.add_argument('--fmax',type=float,default=1100.)
    p.add_argument('--plots',action='store_true',help='キャッシュに解析結果のPNGを作成')
    p.add_argument('--cancel-file',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--clear-cache',action='store_true',help='本ツールのキャッシュだけを削除して終了')
    args=p.parse_args(argv)
    build=None
    lock=None
    try:
        folder=args.folder.resolve()
        if not folder.is_dir(): raise ValueError('音源フォルダが見つかりません')
        lock=folder/'.auto_oto.lock'
        try:
            handle=lock.open('x',encoding='utf-8')
        except FileExistsError:
            lock=None
            raise ValueError('このフォルダは解析中です。終了済みの場合だけ .auto_oto.lock を削除してください')
        handle.write('auto_oto '+__version__);handle.close()
        if args.clear_cache:
            clear(folder);print('キャッシュを削除しました');return 0
        config=AnalysisConfig(fmin=args.fmin,fmax=args.fmax);config.validate()
        files=sorted((x for x in folder.iterdir() if x.is_file() and x.suffix.lower()=='.wav'),key=lambda x:x.name)
        if not files: raise ValueError('フォルダ直下にWAVがありません')
        output=args.output.absolute() if args.output else folder/'oto.ini'
        if output.suffix.lower()!='.ini' or output.name in ('oto_backup.ini','推定後_oto_backup.ini'):
            raise ValueError('出力先はバックアップ名以外の.iniを指定してください')
        if output.is_symlink() or output.resolve().is_relative_to((folder/'キャッシュ').resolve()):
            raise ValueError('リンク先やキャッシュ内には原音設定を保存できません')
        if output.name == 'oto.ini' and output.parent.resolve() == folder:
            initial_oto=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
        else:
            initial_oto=None
        if output.exists() and not args.overwrite:
            raise FileExistsError(f'{output.name} already exists; pass --overwrite to replace it')
        mapping=json.loads(args.name_map.read_text(encoding='utf-8-sig')) if args.name_map else {}
        if not isinstance(mapping,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in mapping.items()):
            raise ValueError('name-map はWAV名と読みの辞書が必要です')
        backend=Backend(args.device)
        print(f'解析デバイス: {backend.device}'+(f' / {backend.reason}' if backend.reason else ''))
        conversions=[]
        for wav in files:
            change=normalization_notice(wav)
            if change:conversions.append(f'{wav.name}: {change}')
        if conversions:
            print('入力WAVは解析用に変換します（元ファイルは変更しません）:\n'+'\n'.join(conversions),file=sys.stderr)
        build=CacheBuild(folder)
        reports=[];errors=[];entries=[];log=[]
        # Short two-mora recordings cannot resolve their own tempo. Use only
        # acoustic estimates from a spread of longer recordings in this folder.
        from .analysis import align
        from .rhythm import fit_grid
        long_files=[]
        for path in files:
            try:
                if len(parse_name(path.name,mapping.get(path.name)).moras)>=6: long_files.append(path)
            except ValueError: pass
        preload={};periods=[]
        if long_files:
            for index in sorted(set(np.linspace(0,len(long_files)-1,min(5,len(long_files))).astype(int))):
                if args.cancel_file and args.cancel_file.exists(): return 3
                path=long_files[index]
                try:
                    parsed=parse_name(path.name,mapping.get(path.name));x,sr=read_wav(path)
                    feature=extract(x,sr,backend,config=config)
                    a,_,score,silent=align(feature,parsed.moras)
                    if silent: continue
                    _,tempo=fit_grid([feature.times[j] for j in a],parsed.moras,feature,score)
                    periods.append(tempo['period_ms']);preload[path.name]=(hashlib.sha256(path.read_bytes()).hexdigest(),feature)
                except (ValueError,OSError,EOFError,RuntimeError):pass
        if len(periods)>=3:
            period=float(np.median(periods))
            if np.median(abs(np.asarray(periods)-period))/period<.1:
                config=replace(config,short_recording_period_ms=period)
                print(f'短い録音の補助BPM: {60000/period:.1f}',flush=True)
        for number,path in enumerate(files,1):
            if args.cancel_file and args.cancel_file.exists(): return 3
            print(f'[{number}/{len(files)}] {path.name}',flush=True)
            try:
                parsed=parse_name(path.name,mapping.get(path.name))
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                x,sr=read_wav(path)
                cached=preload.pop(path.name,None)
                f=cached[1] if cached is not None and cached[0]==digest else extract(x,sr,backend,config=config)
                found,report=analyze(parsed,f,config=config)
                from .preutterance_calibration import calibrate_entries
                found,report['calibration']=calibrate_entries(found,path,cache_dir=build.stage/'音響特徴')
                serialize(found)
                if not found: raise ValueError('ファイル名を確認してください: 生成可能な連続音がありません')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    raise ValueError('解析中にWAVが変更されたため結果を採用しません')
                feature_name=f'音響特徴_{number:04d}.npz'
                np.savez_compressed(build.stage/feature_name,**asdict(f))
                if args.plots:
                    from .plotting import render
                    render(f,found,path.name,build.stage/f'解析結果_{path.stem}.png')
                report.update(wav=path.name,sha256=digest,duration_ms=f.duration_ms,entries=found,
                    features_file=feature_name,device=backend.device)
                reports.append(report);entries.extend(found)
                for e in found:
                    for line in e['line_diagnostics'].values():
                        log.append(f'{path.name} / {e["alias"]} / {line["label"]}: {line["reason"]}')
                print(f'  {len(found)}候補 / 推定BPM {report["tempo"]["bpm"]:.1f}')
            except (ValueError,OSError,EOFError,RuntimeError) as e:
                errors.append(dict(wav=path.name,error=str(e),needs_review=True))
                log.append(f'{path.name}: {e}')
                print(f'  要確認: {e}',file=sys.stderr)
        if args.cancel_file and args.cancel_file.exists(): return 3
        versions={name:importlib.metadata.version(name) for name in ('numpy','soundfile','scipy','librosa','pyworld','matplotlib')}
        document=dict(schema_version=3,algorithm_version=__version__,settings={k:v for k,v in asdict(config).items() if k not in ('overall_review_percent','line_warning_percent')},
            device=backend.device,dependencies=versions,
            reference_policy='runtime acoustic evidence only; no manual oto scores',
            coordinate_system='ms; offset absolute; other oto fields relative to offset; negative cutoff=-(end-offset)',
            files=reports,errors=errors,summary=dict(wavs=len(files),successful_wavs=len(reports),entries=len(entries)))
        (build.stage/'解析結果.json').write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (build.stage/'推定理由.log').write_text('\n'.join(log),encoding='utf-8')
        build.finish(dict(algorithm_version=__version__,settings=asdict(config)))
        if entries:
            if output.name == 'oto.ini' and output.parent.resolve() == folder:
                current=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None
                if current!=initial_oto:raise RuntimeError('oto.ini changed during analysis; refusing to overwrite it')
            write_oto(output,serialize(entries))
        build.publish()
        print(f'キャッシュ: {folder/"キャッシュ"}')
        if entries: print(f'保存: {output} / {len(entries)}候補')
        else: print('生成可能な候補がないため既存の原音設定は変更しません',file=sys.stderr)
        return 2 if errors or not entries else 0
    except (ValueError,OSError,RuntimeError) as e:
        print(f'エラー: {e}',file=sys.stderr);return 1
    finally:
        if build: build.close()
        if lock and lock.exists(): lock.unlink()
