# 軽量CUDA単体版の第三者ソフトウェア

このEXEは、Python 3.12、PyTorch CPU版 2.6.0、CuPy CUDA 12.x版 13.5.1、fastrlock 0.8.3、CUDA 12.8系の実行DLL、および音声解析・画面表示用の第三者ライブラリを含みます。独自コードの利用条件は公開用リポジトリの `LICENSE` を参照してください。第三者の権利には、それぞれのライセンスが適用されます。

- CuPy: MIT。ライセンス文はライセンスZIPの `THIRD_PARTY_LICENSES/cupy/LICENSE`。
- fastrlock: MIT。ライセンス文は `THIRD_PARTY_LICENSES/fastrlock/LICENSE`。
- PyTorch CPU版、Python、NumPy、SciPy、SoundFile、soxr、librosa、PyWorld、Matplotlib、scikit-learn、tkinterdnd2、Tcl/Tkなど: ライセンスZIPの `THIRD_PARTY_LICENSES/` に収録。
- NVIDIAの `cudart64_12.dll`、`cufft64_11.dll`、`nvrtc64_120_0.dll`、`nvrtc-builtins64_128.dll`: [CUDA 12.8の利用・再配布条件](https://docs.nvidia.com/cuda/archive/12.8.0/eula/index.html) を参照してください。このツールはNVIDIAの承認・後援を受けていません。

SoundFileに付属するlibsndfileとPython-SoXRに含まれるlibsoxrにはLGPLの条件が適用されます。対応ソースは [SoundFile 0.14.0](https://pypi.org/project/soundfile/0.14.0/) と [Python-SoXR](https://github.com/dofuuz/python-soxr) を参照してください。利用者がこれらを変更して組み合わせる場合は、公開用リポジトリのソースと `build_cuda_single.py` を使って再ビルドできます。再ビルドには対応するPythonパッケージと配布EXEが必要です。

この通知とライセンスZIPは単体EXEと一緒に提供します。EXE内のCuPyなどのライセンス文は起動時の一時展開先にも含まれますが、閲覧しやすいよう別途まとめています。
