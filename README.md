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
Python、CUDA、PyTorch、sd-scripts、WD14、CLIPはアプリ本体に同梱せず、初回Setupで管理ランタイムへ導入します。

### 1) 事前に用意するもの

- Windows 11 (x64)
- 学習に使えるNVIDIA GPUと最新のNVIDIAドライバー
- 画像を保存するドライブの空き容量
- Managed Training Runtime用に約15 GiB以上の空き容量
- SDXL / Illustrious互換の`.safetensors`ベースモデル
- 学習対象の画像（Characterは8枚以上、Styleは16枚以上を推奨）
- ランタイムやモデルを取得するためのインターネット接続
- ベースモデル・画像・生成物を利用する権利と各モデルのライセンス確認

### 2) ZIPをダウンロードする

GitHub Releasesから対象リリースのWindows版ZIPをダウンロードします。ZIPはGitHub Releasesの公式ページから取得し、ダウンロード完了後に展開してください。

### 3) 展開してWindowsの前提ランタイムを導入する

1. 検証済みのZIPを任意の作業フォルダーへ展開します。`LoRA Factory.exe`をZIPの中から直接起動しないでください。
2. 展開先の`install_vcredist.cmd`をダブルクリックします。
3. UACが表示されたら確認して、Microsoft Visual C++ x64 Redistributableの導入を完了します。

このスクリプトはまずWinGetの公式Microsoftソースを使い、利用できない場合はMicrosoft公式インストーラーを取得して署名を確認してから導入します。コンソールに「already installed」と表示された場合は、すでに導入済みです。導入後にアプリがDLLエラーを表示する場合はWindowsを再起動してから再実行してください。

<https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>

### 4) Git、uv、Codex CLIを準備する

Setupが確認するため、GitとuvをインストールしてPATHへ追加します。PowerShellを閉じて新しく開いた後、次を実行して確認します。

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
git --version
uv --version
```

WinGetを使わない場合は、[Git公式インストーラー](https://git-scm.com/download/win)と[uv公式インストール手順](https://docs.astral.sh/uv/getting-started/installation/)を使用してください。

Runtime Codexを使う場合は、[Codex CLI公式ドキュメント](https://learn.chatgpt.com/docs/codex/cli)に従ってCodex CLIをインストールし、PowerShellで`codex`を実行して`Sign in with ChatGPT`を選びます。ログイン状態を次で確認します。

```powershell
codex --version
codex login status
```

Codex CLIは任意のフォールバックを許可できますが、Codexによるレビューや自動化を使う場合はChatGPTアカウントでログインしてください。認証情報やAPIキーをプロジェクトへコピーする必要はありません。

Runtime Codexは初期状態のCodex CLIとCodexRouter設定を自動判定します。stdout/stderrのpipe詰まりを避け、起動・無通信・総時間を個別に監視し、再試行を1つの総時間予算へ制限します。Routerまたは未知設定で起動に失敗した場合だけ、ユーザー設定を読み込まないisolated profileへ切り替えます。詳しい契約は[Codex CLI実行環境非依存タイムアウト設計](docs/codex-runtime-design.md)を参照してください。

Runtime CodexによるDatasetのCaption / Tag refinementでは、採用された全画像について、metadataを除いた最大辺2048 pxの縮小コピーを最大8枚ずつOpenAIへ送ります。Rawや元画像、元ファイル名、プロジェクトpathは送信しません。各呼び出し後に一時JPEGを削除します。Codexは画像から確認できる不足tagをpin済みWD14語彙から追加でき、無効な追加tagは個別に拒否されます。Trigger Wordは入力値があればその値を保持し、空欄なら検証済みCodex候補の先頭を自動採用します。画像準備またはCodex refinementが失敗した場合は、学習開始前に復旧可能な失敗として停止します。有効な応答でTrigger Word候補だけが3件未満の場合は`AWAITING_REVIEW`で手入力を待ちます。Dataset以外のCodex reviewは既存のフォールバック方針を維持します。画像入力のCLI引数は[Codex CLI reference](https://developers.openai.com/codex/cli/reference/)の`--image`を参照してください。

### 5) 初回起動とSetup

1. 展開先の`LoRA Factory.exe`を起動します。
2. Setup画面で各チェックの詳細を確認します。
3. 初回は「Managed Training Runtime」の`Repair`を押します。
4. ダウンロードとインストールが完了するまで待ちます。PyTorch、ONNX Runtime、固定版sd-scripts、WD14モデル、CLIP画像embeddingモデルが管理ランタイムへ導入されます。
5. NVIDIA GPUの検出とdeep probeが完了し、Runtimeが`READY`になることを確認します。

Setupでは、アプリ内Python 3.12、uv、Git、Codex CLIと認証、NVIDIAドライバー/NVML/GPU、Managed Training Runtime、PyTorch、ONNX CUDA、sd-scriptsを確認します。学習中に使用するGPUは、後でGPU UUIDを選択して明示的に許可します。

### 6) プロジェクトを作成する

`New Project`を開き、次の項目を設定します。

1. `LoRA Name`: 完成する`.safetensors`の名前を入力して`Add Project`を押します。
2. `Preset`: 人物の同一性は`Character`、絵柄は`Style`を選びます。
3. `Trigger Word`: 任意です。入力した固有トークンはそのまま使用し、空欄ならRuntime Codexが検証済み候補から自動設定します。
4. `Base Model`: SDXL / Illustrious互換の`.safetensors`を指定します。
5. `Images / Folders`: 学習画像を複数選択するか、画像フォルダーを指定します。
6. `GPU Pool`: 使用を許可するGPU UUIDだけを選択します。選択していないGPUは使用されません。
7. `Output`: 完成したLoRAを保存する親フォルダーを指定します。

元画像は変更されません。アプリはハッシュを確認してプロジェクト内のRaw Storeへコピーし、加工は作業用コピーに対して行います。入力画像とベースモデルは、利用許諾を確認したものだけを使用してください。

### 7) Datasetを確認して学習を開始する

1. Dataset Reviewでサムネイル、元ファイル名、解像度、判定理由、タグ、最終caption、採用状態を確認します。
2. 低解像度、極端な縦横比、空画像、重複、透かし候補などを確認し、不要な画像を除外します。
3. Characterは8枚以上、Styleは16枚以上の採用画像を用意します。少ない場合は警告や停止になることがあります。
4. `Create LoRA`を押して設定を保存し、検査・前処理・タグ付け・caption生成・学習を開始します。

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

- QtWidgetsの「指定されたプロシージャが見つかりません」エラーは、別アプリのQt DLLが先に解決される環境にも対応しています。解消しない場合は展開先のinstall_vcredist.cmdを再実行し、完了後にWindowsを再起動してください。

- Setupの`uv`または`Git`が赤い: PowerShellを開き直し、`git --version`と`uv --version`を確認してからアプリを再起動します。
- Codexが赤い: `codex login status`を実行し、未ログインなら`codex`から`Sign in with ChatGPT`を実行します。Datasetの画像refinementは復旧後にResumeします。Dataset以外のCodex reviewは、フォールバック許可時に決定論的reviewへ移行できます。
- Managed Training Runtimeが`NOT READY`: Setupの`Repair`を再実行し、約15 GiB以上の空き容量とインターネット接続を確認します。
- GPUが見つからない: NVIDIAドライバーを更新し、`nvidia-smi`が成功することを確認してからアプリを再起動します。
- `Application Services`やDLLエラー: 展開先の`install_vcredist.cmd`を再実行し、完了後にWindowsを再起動します。
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

GitHub公開用の配布ビルドは、Windowsの標準VC++ランタイムDLLを同梱しない次のモードで作成します。

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
