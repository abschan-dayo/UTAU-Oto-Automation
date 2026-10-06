"""Tk stays on the main thread; workers communicate only through a queue."""
import queue
import threading

def choose_methods():
    import tkinter as tk
    from tkinter import ttk
    root=tk.Tk();root.title('実行する処理を選択');root.geometry('580x330')
    frame=ttk.Frame(root,padding=20);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='個別のiniを保存したい方式にチェックを入れてください',font=('Yu Gothic UI',12)).pack(anchor='w',pady=8)
    choices={}
    for method,label in [('existing','① 既存アルゴリズム'),('vision','② Vision（学習済みモデル）'),('alignment','③ Alignment（音素アライメント）')]:
        value=tk.BooleanVar(value=True);choices[method]=value
        ttk.Checkbutton(frame,text=label,variable=value).pack(anchor='w',pady=6)
    hint=tk.StringVar(value='どれを選んでも①②③を解析して中央値を作ります。\n結果は音階フォルダの親に毎回新しいフォルダを作って保存します。')
    ttk.Label(frame,textvariable=hint,wraplength=530).pack(anchor='w',pady=12)
    result=[]
    def start():
        selected=[m for m,v in choices.items() if v.get()]
        if not selected:hint.set('少なくとも1つ選択してください。');return
        result.extend(selected);root.destroy()
    buttons=ttk.Frame(frame);buttons.pack(fill='x')
    ttk.Button(buttons,text='開始',command=start).pack(side='right')
    ttk.Button(buttons,text='キャンセル',command=root.destroy).pack(side='right',padx=8)
    root.mainloop();return result

def launch(operation,methods=None):
    import tkinter as tk
    from tkinter import ttk
    try:
        root=tk.Tk()
    except tk.TclError as error:
        print('進捗画面を開けないため、文字で表示します：'+str(error),flush=True)
        try:
            operation(lambda event:print(event,flush=True));return 0
        except Exception as error:
            print('エラー：'+str(error),flush=True);return 1
    root.title('原音設定ツール — 並列解析の進捗')
    root.geometry('720x470');root.minsize(620,440)
    frame=ttk.Frame(root,padding=22);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='選択した原音設定を生成',font=('Yu Gothic UI',16,'bold')).pack(anchor='w')
    status=tk.StringVar(value='準備中…')
    ttk.Label(frame,textvariable=status,wraplength=640).pack(anchor='w',pady=(12,8))
    activity=ttk.Progressbar(frame,mode='indeterminate');activity.pack(fill='x');activity.start(20)
    widgets={}
    for method,title in [('existing','① 既存'),('vision','② Vision'),('alignment','③ Alignment')]:
        if methods is not None and method not in methods:continue
        text=tk.StringVar(value=title+'：待機中')
        ttk.Label(frame,textvariable=text,wraplength=640).pack(anchor='w',pady=(14,3))
        bar=ttk.Progressbar(frame,maximum=100);bar.pack(fill='x')
        widgets[method]=(title,text,bar,dict(done=0,total=0,errors=0))
    hint=tk.StringVar(value='準備中は時間がかかる場合があります。完了後にこの画面を閉じてください。')
    ttk.Label(frame,textvariable=hint,wraplength=640).pack(anchor='w',pady=12)
    close=ttk.Button(frame,text='閉じる',command=root.destroy,state='disabled');close.pack(anchor='e')
    events=queue.Queue();result=[1];finished=[False]
    def execute():
        try:
            operation(events.put);events.put({'finished':True,'code':0})
        except Exception as error:events.put({'finished':True,'code':1,'error':str(error)})
    def poll():
        for _ in range(500):
            try:event=events.get_nowait()
            except queue.Empty:break
            if 'phase' in event:status.set(event['phase'])
            if event.get('method') in widgets:
                title,text,bar,state=widgets[event['method']];state.update(event)
                percent=100*state['done']/state['total'] if state['total'] else 0
                bar['value']=percent
                text.set(f"{title}：{state['done']}/{state['total']} WAV（{percent:.0f}%） {state.get('message','')}"+ (f" / 失敗 {state['errors']}件" if state['errors'] else ''))
            if event.get('finished'):
                finished[0]=True;result[0]=event['code'];activity.stop();close['state']='normal'
                if event.get('error'):status.set('解析に失敗しました：'+event['error'])
                else:activity.configure(mode='determinate',value=100)
                hint.set('処理が終了しました。「閉じる」で終了できます。')
                return
        root.after(100,poll)
    def request_close():
        if finished[0]:root.destroy()
        else:hint.set('解析中です。保存が終わるまでこの画面を閉じずにお待ちください。')
    root.protocol('WM_DELETE_WINDOW',request_close)
    root.after(100,poll)
    threading.Thread(target=execute,name='oto-coordinator',daemon=False).start()
    root.mainloop();return result[0]
