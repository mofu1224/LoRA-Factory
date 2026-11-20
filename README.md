---
type: 利用ガイド
---

# LoRA Factory

LoRA Factory は、Windows 11 向けのGUIツールです。  
手元の画像から **SDXL / Illustrious 互換のLoRA** を作成する流れを一元化します。

- 画像取り込み
- 検査・前処理
- WD14 タグ付け
- 学習実行
- LoRA の比較・採点・輸出

---

## 使い方（ダウンロードからLoRA完成まで）

以下はGitHub ReleasesからWindows版ZIPをダウンロードした利用者向けの手順です。
PythonはZIPに含まれ、uvとGitは初回Setupでアプリフォルダー内に導入されます。
公開ZIPはWindows側のMicrosoft Visual C++ランタイム（x64）を使用します。未導入の場合は
同梱の`install_vcredist.cmd`を実行してください。この操作には管理者権限が必要です。
PyTorch、sd-scripts、WD14、CLIPはアプリ本体に同梱せず、初回Setupでアプリフォルダー内の
管理ランタイムへ導入します（初回のみインターネット接続と約15 GiBの空き容量が必要）。
設定、モデルキャッシュ、学習環境、生成物はすべてアプリフォルダー内に保存されるため、
フォルダーを丸ごとコピーすれば別の場所や別のPCでも同じ手順で利用できます。

### 1) 事前に用意するもの

- Windows 11 (x64)
- 学習に使えるNVIDIA GPUと最新のNVIDIAドライバー
- 画像を保存するドライブの空き容量
- Managed Training Runtime用に約15 GiB以上の空き容量
- SDXL / Illustrious互換の`.safetensors`ベースモデル
- 学習対象の画像（Characterは8枚以上、Styleは16枚以上を推奨）
- 初回Setupで固定版ツールとランタイムを取得するためのインターネット接続
- ベースモデル・画像・生成物を利用する権利と各モデルのライセンス確認

Python、Git、uv、Codex CLIの事前インストールは不要です。

### 2) ZIPをダウンロードする

[リリース一覧](https://github.com/mofu1224/LoRA-Factory/releases)で対象バージョンの **Assets** を開き、ソースコードではなくWindows版ZIPを選びます。`.sha256` が添付されている場合は、展開前に `Get-FileHash -Algorithm SHA256 <ZIPのパス>` の結果と照合してください。

### 3) 展開して起動する

1. 検証済みのZIPを任意の作業フォルダーへ展開します。`LoRA Factory.exe`をZIPの中から直接起動しないでください。
2. 展開先の`LoRA Factory.exe`を起動します。

パスに空白や日本語を含む、書き込み可能なフォルダーに展開してください。

### 4) 初回起動時に固定版ツールを準備する

初回起動またはSetupの`Repair`で、アプリはpin留めした固定版の`uv`と`git`を公式リリースから
アプリフォルダー内の`tools\`へ取得し、SHA-256を検証してから使用します。PATH上の`uv`や`git`は
必要ありません。取得できたツールは次回以降そのまま再利用されます。

Runtime Codexを使う場合だけ、任意で[Codex CLI公式ドキュメント](https://learn.chatgpt.com/docs/codex/cli)に従ってCodex CLIをインストールし、PowerShellで`codex`を実行して`Sign in with ChatGPT`を選びます。Codexは必須ではなく、未導入でも他の機能は動作します。

Codex CLIは任意のフォールバックを許可できますが、Codexによるレビューや自動化を使う場合はChatGPTアカウントでログインしてください。認証情報やAPIキーをプロジェクトへコピーする必要はありません。

Runtime Codexは初期状態のCodex CLIとCodexRouter設定を自動判定します。stdout/stderrのpipe詰まりを避け、起動・無通信・総時間を個別に監視し、再試行を1つの総時間予算へ制限します。Routerまたは未知設定で起動に失敗した場合だけ、ユーザー設定を読み込まないisolated profileへ切り替えます。詳しい契約は[Codex CLI実行環境非依存タイムアウト設計](docs/codex-runtime-design.md)を参照してください。

Runtime CodexによるDatasetのCaption / Tag refinementでは、採用された全画像について、metadataを除いた最大辺2048 pxの縮小コピーを最大8枚ずつOpenAIへ送ります。Rawや元画像、元ファイル名、プロジェクトpathは送信しません。各呼び出し後に一時JPEGを削除します。Codexは画像から確認できる不足tagをpin済みWD14語彙から追加でき、無効な追加tagは個別に拒否されます。既定の`AUTO`では各batchを上限付きで再試行し、timeout・未インストール・不正な構造化応答は、検証済みの元caption/tagを保つ決定論的fallbackへ切り替えます。安全な提案は自動採用し、それ以外は元captionへ戻して承認待ちを発生させません。Trigger Wordは入力値を保持し、空欄で候補が得られない場合も安定したローカルトークンを自動生成します。明示的な`REVIEW`、意味tagを満たせない入力、画像を安全に準備できない入力だけは`AWAITING_REVIEW`または復旧可能な停止として学習前に止まります。Dataset以外のCodex reviewは既存のフォールバック方針を維持します。画像入力のCLI引数は[Codex CLI reference](https://developers.openai.com/codex/cli/reference/)の`--image`を参照してください。

### 5) 初回起動とSetup

1. 展開先の`LoRA Factory.exe`を起動します。
2. Setup画面で各チェックの詳細を確認します。
3. 初回は「Managed Training Runtime」の`Repair`を押します。
4. ダウンロードとインストールが完了するまで待ちます。PyTorch、ONNX Runtime、固定版sd-scripts、WD14モデル、CLIP画像embeddingモデルが管理ランタイムへ導入されます。
5. NVIDIA GPUの検出とdeep probeが完了し、Runtimeが`READY`になることを確認します。

Setupでは、アプリ内Python 3.12、アプリが`tools\`へ導入するuvとGit、任意のCodex CLIと認証、NVIDIAドライバー/NVML/GPU、Managed Training Runtime、PyTorch、ONNX CUDA、sd-scriptsを確認します。学習中に使用するGPUは、後でGPU UUIDを選択して明示的に許可します。

### 6) プロジェクトを作成する

`New Project`を開き、次の項目を設定します。

1. `LoRA Name`: 完成する`.safetensors`の名前を入力して`Add Project`を押します。
2. `Preset`: 人物の同一性は`Character`、絵柄は`Style`を選びます。
3. `Trigger Word`: 任意です。入力した固有トークンはそのまま使用し、空欄でもAUTOではCodex候補または安定したローカルトークンを自動設定します。
4. `Base Model`: SDXL / Illustrious互換の`.safetensors`を指定します。
5. `Images / Folders`: 学習画像を複数選択するか、画像フォルダーを指定します。
6. `GPU Pool`: 使用を許可するGPU UUIDだけを選択します。選択していないGPUは使用されません。
7. `Output`: 完成したLoRAを保存する親フォルダーを指定します。

元画像は変更されません。アプリはハッシュを確認してプロジェクト内のRaw Storeへコピーし、加工は作業用コピーに対して行います。入力画像とベースモデルは、利用許諾を確認したものだけを使用してください。

### 7) Datasetを確認して学習を開始する

1. Dataset Reviewでサムネイル、元ファイル名、解像度、判定理由、タグ、最終caption、採用状態を確認します。
2. 低解像度、極端な縦横比、空画像、重複、透かし候補などを確認し、不要な画像を除外します。
3. Characterは8枚以上、Styleは16枚以上の採用画像を用意します。少ない場合は警告や停止になることがあります。
4. `Create LoRA`を押して設定を保存し、検査・前処理・タグ付け・caption生成・学習を開始します。AUTOの保留runは次回GUI起動時にも安全に再開します。

学習中はstage、epoch/step、loss、使用GPU、checkpoint、sample progressを確認できます。停止したい場合は`Cancel`を使ってください。完成済みartifactとresume stateを残して安全に停止し、`Recent Projects`から再開できます。学習中はアプリを終了せず、ディスクの空き容量も確保してください。

### 8) 完成したLoRAを確認する

Completion画面で推奨weight、preview、comparison、alternatives、出力先を確認します。完成フォルダーには通常、次のファイルが含まれます。

```text
<LoRAName>.safetensors
alternatives/
preview.png
comparison.png
README.txt
training_info.json
evaluation.json
resolved_config.yaml
reproducibility_manifest.json
```

SettingsでAUTOMATIC1111、Forge、ComfyUIのrootまたはLoRAフォルダーを登録すると、完成したLoRAをコピーできます。同名ファイルは上書きせず、コピー後にハッシュを確認します。

### 9) よくある問題

- QtWidgetsやDLLのエラーで起動できない: 同梱の`install_vcredist.cmd`でx64ランタイムを導入または修復してから、もう一度起動してください。

- Setupの`uv`または`Git`が赤い: Setupの`Repair`を実行し、初回の固定版ツール取得に必要なインターネット接続を確認します。取得後はアプリフォルダー内の`tools\`から使用され、PATH上のインストールは不要です。
- Codexが赤い: 任意機能です。利用する場合は`codex login status`を実行し、未ログインなら`codex`から`Sign in with ChatGPT`を実行します。AUTOではDatasetのtimeout・不正応答を元tagへ戻して自動継続します。明示的REVIEWまたは意味tagを満たせない入力はDataset Reviewで確認してください。Dataset以外のCodex reviewは、フォールバック許可時に決定論的reviewへ移行できます。
- Managed Training Runtimeが`NOT READY`: Setupの`Repair`を再実行し、約15 GiB以上の空き容量とインターネット接続を確認します。
- アプリフォルダーを移動・改名した: 設定や管理ランタイムの絶対パスが変わるため、Setupで`Managed Training Runtime`の`Repair`を実行します。ダウンロード済みのsd-scriptsとモデルは再利用され、Python環境だけが現在の場所に合わせて再構築されます。
- GPUが見つからない: NVIDIAドライバーを更新し、`nvidia-smi`が成功することを確認してからアプリを再起動します。
- `Application Services`やDLLエラー: まずWindowsを再起動します。解消しない場合のみ、同梱の`install_vcredist.cmd`を任意で実行します。
- ベースモデルが拒否される: SDXL / Illustrious互換の`.safetensors`であることと、モデルのライセンスを確認します。
- ディスク不足: Managed Training Runtimeとプロジェクト出力先を別ドライブへ移すか、十分な空き容量を確保します。

詳しい画面仕様、Datasetルール、cancel/resume、診断方法は[日本語ユーザーガイド](./docs/user-guide-ja.md)を参照してください。

### 10) 開発者（ソース実行）

```powershell
.\scripts\bootstrap.ps1
.\scripts\run_dev.ps1
```

必要に応じて `-Console` を付けてコンソール表示付きで起動します。

---

## 設計の前提

- `dataset/raw/` は原則変更しない（破壊しない）
- 画像は検証付きコピーで管理し、派生データとして保存
- DatasetのRuntime Codex refinementでは、採用画像のmetadataを除いた縮小コピーだけをOpenAIへ送る
- 監査ログや診断情報は最小化し、外部に秘匿情報が残らないよう制御

---

## ポータブル構成

アプリフォルダーは次の構成を持ちます。初回起動後に生成されるものは、いずれもアプリフォルダー
直下に作られ、PC本体のユーザー領域（`%LOCALAPPDATA%`など）は使いません。

```text
LoRA Factory/           配布物（ZIPを展開したフォルダー）
├─ LoRA Factory.exe     実行ファイル
├─ _internal/           アプリ本体、アプリ内Python、Qt
├─ tools/               初回起動時に取得する固定版 uv / git（初回のみ通信）
├─ data/                settings.json、codex-scratch（初回起動後に生成）
├─ runtime/             Managed Training Runtime（PyTorch/sd-scripts/WD14/CLIP、初回Setup）
├─ cache/               uv/Hugging Face/PyTorch のキャッシュ
├─ python/              uvが管理するPython本体
└─ Project/             作成したLoRAプロジェクト
```

フォルダーを丸ごとコピーすれば、別の場所や別のPCでも設定と学習環境を保ったまま利用できます。
同じ絶対パス以外へ移動した場合は、Setupの`Repair`でPython環境だけを現在の場所に合わせて
再構築します（ダウンロード済みのsd-scriptsとモデルは再利用）。

初回に通信が必要なのは次の2つです。いずれも検証済みの固定版を取得します。

1. 初回起動時の`uv`と`git`（アプリフォルダー内`tools\`へ、SHA-256検証つき）
2. 初回SetupのManaged Training Runtime（約15 GiBをアプリフォルダー内`runtime\`へ）

固定版のバージョンとハッシュは`src/lora_factory/runtime/toolchain.py`と`backend-manifest.json`で
管理しています。

---

## 主なチェックコマンド

```powershell
uv run --frozen lora-factory doctor
uv run --frozen lora-factory fake-e2e
uv run --frozen lora-factory inspect-model "D:\Models\base.safetensors"
```

---

## リリースとビルド

```powershell
.\scripts\build_windows.ps1
```

既定のポータブルビルドは、Microsoft Visual C++ランタイムDLLとアプリ内CPythonを
アプリフォルダーへ同梱し、ビルド時に両方の存在を検証します。Windowsのユーザー領域や
システムへの導入なしで起動できます。

GitHub Releases用のビルドは、標準のVC++ランタイムDLLを同梱しない次のモードで作成します。

```powershell
.\scripts\build_windows.ps1 -SystemVcRuntime
```

`dist\windows-<timestamp>\LoRA Factory\` に one-dir 版を出力します。  
`licenses\`、QtのLGPL/GPL全文、Qt配布手順、Microsoft Visual C++ランタイム通知を
同梱します。配布前に `THIRD_PARTY_NOTICES.md` と `licenses\` を確認してください。

---

## 参考情報

- [docs/user-guide-ja.md](./docs/user-guide-ja.md)
- [docs/developer-guide.md](./docs/developer-guide.md)
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

## ライセンス

本体は [MIT License](./LICENSE) です。

PySide6/Qt、CPython、PyInstaller、各Pythonパッケージ、Microsoft Visual C++ランタイム、
CUDA、モデル、ユーザーが選択するベースモデルはそれぞれの条件に従います。
