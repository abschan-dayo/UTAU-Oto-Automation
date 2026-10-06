# 原音設定推定アルゴリズム

このフォルダには推定処理のソース、CLIランチャー、自動推定ツールの操作画面と保存処理を置いています。波形を手動編集する原音設定エディタのソースは含みません。Vision・Alignment推定器のコードは含みますが、学習済みモデルはソース配布に含みません。利用条件は上位の `LICENSE` を参照してください。

## 出力

既定では、指定した音源フォルダに `推定後_oto.ini` を作成します。既存の `oto.ini` は出力先として使いません。同名の推定結果が既にある場合は何も変更せず停止します。上書きする場合だけ `--overwrite` を明示してください。上書き時は最初の旧推定結果を `推定後_oto_backup.ini` に保存し、その後はこのバックアップを保持します。

## 実行方法

Python 3.12 と `requirements.txt` 記載の依存ライブラリを用意したうえで、次のように実行します。

Vision・Alignment の推定コードを利用する場合は、対象モデルと PyTorch 2.6.0 も必要です。モデルはソースリポジトリに置いていません。Windows配布版は、監査済みの公開用モデルをEXEに含めます。

```powershell
python auto_oto.py "<VOICEBANK_DIR>"
```

デバイスを指定する場合:

```powershell
python auto_oto.py "<VOICEBANK_DIR>" --device cpu
python auto_oto.py "<VOICEBANK_DIR>" --device cuda
python auto_oto.py "<VOICEBANK_DIR>" --overwrite
```

自動推定ツールの操作画面は `pythonw v1_oto.pyw` で起動できます。Vision・Alignment方式を使う場合は、公式Releaseの配布物に含まれる公開用4モデルを、このフォルダ直下の `models/` に置いてください。`models/` はGitに登録しません。CPU版とCUDA版のビルドにはそれぞれ対応するPyTorch環境が必要です。

ファイル名からエイリアスを推定できない場合は、必要に応じてUTF-8のJSONマップを `--name-map` で指定できます。実際に使うWAV名やマップは公開リポジトリへ追加しないでください。

## データの扱い

入力WAVと設定ファイルは、公開用ソースやIssueへ含めないでください。実行時に作られるキャッシュ、レポート、ログはローカルの音源フォルダ側に保存されます。共有する前に音源名やパス等の識別情報がないか確認してください。
