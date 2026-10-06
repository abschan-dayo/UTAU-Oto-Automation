"""Small v1 Windows workflow; the advanced editor is deliberately excluded."""
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk,filedialog,messagebox

from .v1_service import check_folder,clear_analysis_cache,generate,oto_state
from .vlabeler import saved_executable,save_executable,open_in_vlabeler
from .runtime import default_device
from .v1_log import OperationLog

METHODS=(
    ('existing','1. スペクトログラム・音響解析','スペクトログラムや音量・周波数などの音響特徴を解析し、音の変化から原音設定位置を推定します。'),
    ('vision','2. スペクトログラム画像認識','スペクトログラムをCNNで画像解析し、学習した境界パターンから先行発声位置を推定します。'),
    ('alignment','3. 音素位置解析','音声の特徴変化を音素列と照合し、発音中の各音素の位置を推定して原音設定の境界を決定します。'),
)
STEPS=('WAV読込中','特徴抽出中','境界解析中','統合中','推定後_oto.ini生成中')


def system_light():
    if sys.platform=='darwin':
        try:
            result=subprocess.run(['defaults','read','-g','AppleInterfaceStyle'],
                                  capture_output=True,text=True,timeout=2)
            return result.returncode!=0 or result.stdout.strip().lower()!='dark'
        except (OSError,subprocess.TimeoutExpired):return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as key:
            value,_=winreg.QueryValueEx(key,'AppsUseLightTheme')
            return bool(value)
    except (OSError,ImportError):return True


def apply_theme(root):
    light=system_light()
    bg='#f7f8fb' if light else '#101114'
    fg='#172033' if light else '#f4f5f7'
    card='#ffffff' if light else '#20232a'
    accent='#3268d4' if light else '#87acff'
    font='Hiragino Sans' if sys.platform=='darwin' else 'Yu Gothic UI'
    root.configure(bg=bg)
    style=ttk.Style(root)
    style.theme_use('clam')
    style.configure('TFrame',background=bg)
    style.configure('Card.TFrame',background=card)
    style.configure('TLabel',background=bg,foreground=fg,font=(font,10))
    style.configure('Card.TLabel',background=card,foreground=fg,font=(font,10))
    style.configure('Title.TLabel',background=bg,foreground=fg,font=(font,17,'bold'))
    style.configure('TCheckbutton',background=card,foreground=fg,font=(font,11))
    style.map('TCheckbutton',background=[('active',card)],foreground=[('active',fg)])
    style.configure('TButton',padding=(13,8),font=(font,10))
    style.configure('Horizontal.TProgressbar',troughcolor=card,background=accent)
    return bg,fg,card,accent


class App:
    def __init__(self,root,initial=None):
        self.root=root;self.folder=None;self.cancel=threading.Event();self.events=queue.Queue();self.log_path=None
        root.title('UTAU原音設定自動化ツール');root.geometry('760x510');root.minsize(650,470)
        self.bg,self.fg,self.card,self.accent=apply_theme(root)
        self.theme_is_light=system_light();self.progress_window=None
        body=ttk.Frame(root,padding=24);body.pack(fill='both',expand=True)
        ttk.Label(body,text='原音設定を生成',style='Title.TLabel').pack(anchor='w')
        self.folder_text=tk.StringVar(value='WAVが入った音階フォルダを選択してください')
        path_row=ttk.Frame(body);path_row.pack(fill='x',pady=(18,14))
        ttk.Label(path_row,textvariable=self.folder_text,wraplength=570).pack(side='left',fill='x',expand=True)
        ttk.Button(path_row,text='フォルダを選択',command=self.pick_folder).pack(side='right')
        self.drop=ttk.Label(body,text='フォルダをここへドラッグ＆ドロップ',anchor='center')
        self.drop.pack(fill='x',ipady=15,pady=(0,14))
        self._drop_support()
        ttk.Label(body,text='解析方式を選択').pack(anchor='w',pady=(0,8))
        self.flags={}
        for key,title,detail in METHODS:
            card=ttk.Frame(body,style='Card.TFrame',padding=(12,8));card.pack(fill='x',pady=3)
            flag=tk.BooleanVar(value=True);self.flags[key]=flag
            ttk.Checkbutton(card,text=title,variable=flag).pack(anchor='w')
            ttk.Label(card,text=detail,style='Card.TLabel',wraplength=680).pack(anchor='w',padx=24)
        row=ttk.Frame(body);row.pack(fill='x',pady=(18,0))
        self.clear_button=ttk.Button(row,text='キャッシュを削除',command=self.clear_cache)
        self.clear_button.pack(side='left')
        ttk.Button(row,text='ログを開く',command=self.open_log).pack(side='left',padx=(8,0))
        self.start=ttk.Button(row,text='解析を開始',command=self.start_analysis)
        self.start.pack(side='right')
        if initial:self.set_folder(initial)
        root.after(3000,self._refresh_theme)

    def _refresh_theme(self):
        if system_light()!=self.theme_is_light:
            self.theme_is_light=system_light()
            self.bg,self.fg,self.card,self.accent=apply_theme(self.root)
            if self.progress_window is not None and self.progress_window.winfo_exists():
                self.progress_window.configure(bg=self.bg)
        self.root.after(3000,self._refresh_theme)

    def _drop_support(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind('<<Drop>>',lambda e:self.set_folder(self.root.tk.splitlist(e.data)[0]))
        except (ImportError,tk.TclError,AttributeError):
            self.drop.configure(text='フォルダを選択するか、フォルダを実行ファイルへドロップしてください')

    def pick_folder(self):
        path=filedialog.askdirectory(parent=self.root)
        if path:self.set_folder(path)

    def set_folder(self,path):
        try:self.folder,_=check_folder(path)
        except (ValueError,OSError) as error:
            messagebox.showerror('音源フォルダを確認',str(error),parent=self.root);return
        self.folder_text.set(str(self.folder))

    def clear_cache(self):
        if self.folder is None:
            messagebox.showwarning('音源フォルダを選択','先に音源フォルダを選択してください。',parent=self.root);return
        if not messagebox.askyesno('キャッシュを削除',
                '選択した音源フォルダの再生成可能な解析キャッシュを削除しますか？\nWAVとoto.iniは変更しません。',parent=self.root):return
        try:
            log=OperationLog('キャッシュ削除');self.log_path=log.path
            log.write(f'音源フォルダ：{self.folder}')
            removed=clear_analysis_cache(self.folder)
            log.write('結果：キャッシュを削除しました。' if removed else '結果：削除するキャッシュはありません。')
            messagebox.showinfo('キャッシュ削除',
                                'キャッシュを削除しました。' if removed else '削除するキャッシュはありません。',parent=self.root)
        except Exception as error:
            if 'log' in locals():log.write(f'失敗：{error}')
            messagebox.showerror('キャッシュを削除できません',str(error),parent=self.root)

    def open_log(self):
        if self.log_path is None or not self.log_path.is_file():
            messagebox.showinfo('ログ','この起動中のログはまだありません。',parent=self.root);return
        try:
            if sys.platform=='win32':os.startfile(self.log_path)
            elif sys.platform=='darwin':subprocess.Popen(['open',str(self.log_path)])
            else:subprocess.Popen(['xdg-open',str(self.log_path)])
        except OSError as error:messagebox.showerror('ログを開けません',str(error),parent=self.root)

    def start_analysis(self):
        if self.folder is None:
            messagebox.showwarning('音源フォルダを選択', 'WAVが入った音階フォルダを選択してください。',parent=self.root);return
        methods=tuple(k for k,flag in self.flags.items() if flag.get())
        if not methods:
            messagebox.showwarning('解析方式を選択','解析方式を1つ以上選択してください。',parent=self.root);return
        try:existing=(self.folder/'推定後_oto.ini').exists()
        except (ValueError,OSError) as error:
            messagebox.showerror('oto.iniを確認',str(error),parent=self.root);return
        overwrite=False
        if existing:
            choice=messagebox.askyesnocancel('既存の推定結果','既存の推定後_oto.iniを上書きしますか？\n上書き前のファイルは推定後_oto_backup.iniへ退避します。',parent=self.root)
            if choice is not True:return
            overwrite=True
        self._progress(methods,overwrite,existing)

    def _progress(self,methods,overwrite,existing):
        window=tk.Toplevel(self.root);window.title('原音設定を生成しています');window.geometry('680x470')
        self.progress_window=window
        window.transient(self.root);window.grab_set();apply_theme(window)
        body=ttk.Frame(window,padding=24);body.pack(fill='both',expand=True)
        heading=tk.StringVar(value='WAV読込中')
        ttk.Label(body,textvariable=heading,style='Title.TLabel').pack(anchor='w')
        percent=tk.StringVar(value='0%')
        ttk.Label(body,textvariable=percent).pack(anchor='e')
        bar=ttk.Progressbar(body,maximum=100);bar.pack(fill='x',pady=(4,18))
        current=tk.StringVar(value='準備中')
        ttk.Label(body,textvariable=current,wraplength=620).pack(anchor='w')
        steps={}
        for step in STEPS:
            var=tk.StringVar(value='○ '+step.replace('中',''))
            ttk.Label(body,textvariable=var).pack(anchor='w',pady=4);steps[step]=var
        count=tk.StringVar(value='処理済みWAV 0 / 0')
        ttk.Label(body,textvariable=count).pack(anchor='w',pady=(14,4))
        active=tk.StringVar(value='使用中の解析方式：準備中')
        ttk.Label(body,textvariable=active).pack(anchor='w')
        cancel_button=ttk.Button(body,text='中止',command=self.cancel.set)
        cancel_button.pack(anchor='e',pady=(18,0))
        self.cancel.clear();self.start['state']='disabled';self.clear_button['state']='disabled'
        workers=('existing',)+tuple(m for m in methods if m!='existing')
        totals={m:(0,0,'') for m in workers}
        try:
            log=OperationLog('原音設定生成');self.log_path=log.path
            device=default_device()
            log.write(f'音源フォルダ：{self.folder}')
            log.write('解析方式：'+', '.join(methods))
            log.write('解析デバイス指定：'+device.upper())
        except Exception as error:
            self.start['state']='normal';self.clear_button['state']='normal';window.destroy()
            messagebox.showerror('ログを作成できません',str(error),parent=self.root);return
        def progress(event):
            if 'phase' in event:log.write('進捗：'+str(event['phase']))
            if event.get('message'):log.write(str(event.get('method','解析'))+'：'+str(event['message']))
            self.events.put(event)
        def worker():
            try:
                if device=='cuda':
                    from .features import Backend
                    backend=Backend('cuda')
                    log.write(f'CUDA動作確認：{backend.backend_name}（画像モデルはCPU実行の場合あり）')
                result=generate(self.folder,methods,overwrite,progress,self.cancel,device=device)
                log.write(f'生成完了：{result.path} / {result.entries}設定')
                self.events.put(('done',result))
            except Exception as error:
                log.write('失敗：'+str(error))
                log.write(traceback.format_exc())
                self.events.put(('error',error))
        thread=threading.Thread(target=worker,daemon=False);thread.start()
        def poll():
            while True:
                try:event=self.events.get_nowait()
                except queue.Empty:break
                if isinstance(event,tuple):
                    self.start['state']='normal';self.clear_button['state']='normal';cancel_button['state']='disabled';window.destroy()
                    if event[0]=='error':
                        error=event[1]
                        if isinstance(error,InterruptedError):messagebox.showinfo('解析を中止','既存の原音設定は変更していません。',parent=self.root)
                        else:messagebox.showerror('解析できませんでした',str(error),parent=self.root)
                    else:self._finish(event[1])
                    return
                if 'phase' in event:heading.set(event['phase'])
                if 'step' in event:
                    step=event['step'];heading.set(step)
                    for item,var in steps.items():var.set(('● ' if item==step else '○ ')+item.replace('中',''))
                method=event.get('method')
                if method in totals:
                    totals[method]=(event.get('done',totals[method][0]),event.get('total',totals[method][1]),event.get('message',''))
                    label=next(title for key,title,_ in METHODS if key==method)
                    active.set('使用中の解析方式：'+label)
                    current.set('現在処理中のWAV：'+totals[method][2])
                done=sum(value[0] for value in totals.values());total=sum(value[1] for value in totals.values())
                percent.set(f'{done/total*100:.0f}%' if total else '0%')
                bar['value']=done/total*100 if total else 0
                count.set(f'処理済みWAV {done} / {total}')
            window.after(100,poll)
        window.protocol('WM_DELETE_WINDOW',self.cancel.set)
        window.after(100,poll)

    def _finish(self,result):
        details=(f'既存oto.ini：{"あり" if result.source_had_oto else "なし"}\n'
                 f'保存方法：{"推定結果を上書き" if result.overwritten else "新規作成"}\n'
                 f'保存先：{result.path}\n'
                 f'設定数：{result.entries}\n'
                 f'推定後_oto_backup.ini：{"新規作成" if result.backup_created else "既存のものを保持" if result.overwritten else "作成なし"}')
        messagebox.showinfo('原音設定を生成しました',details,parent=self.root)
        if messagebox.askyesno('vLabeler','vLabelerで編集を続けますか？',parent=self.root):
            executable=saved_executable()
            if executable is None:
                if sys.platform=='darwin':
                    chosen=filedialog.askopenfilename(title='vLabeler.appを選択',filetypes=[('macOSアプリ','*.app')],parent=self.root)
                else:
                    chosen=filedialog.askopenfilename(title='vLabeler.exeを選択',filetypes=[('実行ファイル','*.exe')],parent=self.root)
                if not chosen:return
                try:save_executable(chosen);executable=Path(chosen)
                except (ValueError,OSError) as error:
                    messagebox.showerror('vLabelerを登録できません',str(error),parent=self.root);return
            try:open_in_vlabeler(executable,self.folder,result.path)
            except Exception as error:messagebox.showerror('vLabelerを開けません',str(error),parent=self.root)


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root=TkinterDnD.Tk()
    except ImportError:root=tk.Tk()
    initial=sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith('-psn_') else None
    app=App(root,initial)
    if sys.platform=='darwin':
        root.createcommand('::tk::mac::OpenDocument',
                           lambda *paths:app.set_folder(paths[0]) if paths else None)
    root.mainloop()
