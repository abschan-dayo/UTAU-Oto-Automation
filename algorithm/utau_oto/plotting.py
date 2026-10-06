"""Optional development plots, no GUI or HTML output."""
import numpy as np


def render(features,entries,wav_name,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={font.name for font in font_manager.fontManager.ttflist}
    for font in ('Meiryo','Yu Gothic','Noto Sans CJK JP','IPAexGothic'):
        if font in available:
            plt.rcParams['font.family']=font;break
    plt.rcParams['axes.unicode_minus']=False
    f=features
    fig,axes=plt.subplots(4,1,figsize=(17,11),sharex=True,
        gridspec_kw={'height_ratios':[1,1,3,1]},layout='constrained')
    fig.suptitle(wav_name+' / 解析結果',fontsize=13)
    axes[0].fill_between(f.times,-f.peak,f.peak,color='#cfd8dc')
    axes[0].set_ylabel('波形ピーク')
    axes[1].plot(f.times,f.power_db,color='gold',linewidth=1)
    axes[1].axhline(f.noise_floor_db,color='gray',linestyle=':')
    axes[1].set_ylabel('Power / dB')
    freq=f.frequencies_hz
    axes[2].pcolormesh(f.times,freq[freq<6000],f.spectrum_db[:,freq<6000].T,
        shading='auto',cmap='magma',vmin=float(np.max(f.spectrum_db))-75,
        vmax=float(np.max(f.spectrum_db)))
    axes[2].set_ylabel('スペクトログラム / Hz')
    axes[3].plot(f.times,np.where(f.world_f0>0,f.world_f0,np.nan),color='orange',label='WORLD')
    axes[3].plot(f.times,np.where(f.f0>0,f.f0,np.nan),color='gray',alpha=.6,label='自己相関')
    axes[3].set_ylabel('F0 / Hz');axes[3].set_xlabel('WAV先頭からの時間 / ms');axes[3].legend(loc='upper right')
    colors=dict(offset='yellow',overlap='lime',preutterance='red',consonant='dodgerblue',cutoff='white')
    for ax in axes: ax.set_facecolor('#191919')
    for n,e in enumerate(entries):
        for key,diag in e['line_diagnostics'].items():
            t=diag['selected_absolute_ms']
            for ax in axes: ax.axvline(t,color=colors[key],linewidth=.65,alpha=.85)
            if key=='preutterance':
                for c in diag['candidates']:
                    if not c['selected']:
                        axes[2].scatter(c['absolute_ms'],5500,c='red',s=8,alpha=.4)
                selected=next(c for c in diag['candidates'] if c['selected'])
                text=e['alias']
                axes[0].text(t,.94-(n%2)*.35,text,transform=axes[0].get_xaxis_transform(),
                    fontsize=7,color='white',va='top',bbox=dict(facecolor='black',alpha=.7,edgecolor='none',pad=2))
    axes[0].set_title('黄: 左ブランク / 緑: オーバーラップ / 赤: 先行発声 / 青: 固定範囲 / 白: 右ブランク',fontsize=9)
    import io
    from .writer import atomic_write
    try:
        buf=io.BytesIO()
        fig.savefig(buf,format='png',dpi=115)
        atomic_write(path,buf.getvalue(),overwrite=True)
    finally:
        plt.close(fig)
