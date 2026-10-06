# 原音設定推定アルゴリズム

このフォルダーには原音設定の解析アルゴリズム、CLI、実行・保存処理を含みます。手動編集用の原音設定エディタのソースは含みません。利用条件はリポジトリ直下の `LICENSE` を参照してください。

## GUI出力

GUIは選択した音源フォルダーの `oto.ini` に推定結果を保存します。既存ファイルがある場合は画面で確認し、「はい」の場合だけ置き換えます。初回の上書き前に `oto_backup.ini` を作成し、以後は既存バックアップを保持します。解析中にoto.iniが変更された場合は保存を中止します。

## CLI出力

既定では音源フォルダーの `oto.ini` に保存します。既存ファイルを上書きする場合は `--overwrite` を指定してください。初回は `oto_backup.ini` へ退避します。

```powershell
cd algorithm
python auto_oto.py "<VOICEBANK_DIR>"
python auto_oto.py "<VOICEBANK_DIR>" --overwrite
```

別の `.ini` ファイルを指定する場合は `--output` を使えます。ファイル名解析に使うカスタムマップはUTF-8 JSONで用意し、実データを公開リポジトリへ追加しないでください。

## モデル

学習済みモデルはリポジトリに含めません。配布物の `_internal/models/` から必要なモデルをローカルの `algorithm/models/` に配置します。`models/` はGit対象外です。
