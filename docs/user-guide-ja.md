# LoRA Factory 日本語ユーザーガイド

## 1. 起動

GitHub ReleasesのWindows ZIPを使う場合は、ZIPとSHA-256 fileをdownloadしてhashを確認し、展開後の`LoRA Factory.exe`を起動します。先に[Codex CLI公式ドキュメント](https://learn.chatgpt.com/docs/codex/cli)に従ってCLIをinstallし、PowerShellで`codex`を実行して`Sign in with ChatGPT`を選びます。

source checkoutから起動する場合だけ、初回にPowerShellで次を実行します。

```powershell
.\scripts\bootstrap.ps1
```

通常起動は次です。黒いconsole windowは常駐しません。

```powershell
.\scripts\run_dev.ps1
```

開発中のerrorを確認したい場合は`.\scripts\run_dev.ps1 -Console`を使います。

## 2. Setup

Setup画面でPython 3.12、uv、Git、Codex CLIの認証、NVIDIA driver/NVML/GPU、managed training runtime、PyTorch、ONNX CUDA、sd-scripts、任意の出力先を確認します。Real Backendを初めて使う場合は「Managed Training Runtime」の`Repair`を押してください。runtimeのinstallと選択可能GPU上のdeep probeはbackgroundで実行され、GUIは固まりません。runtimeにはPyTorch、ONNX Runtime、pin済みsd-scripts、WD14 modelが入り、15 GiB以上の空き容量を見込んでください。

Runtime Codexは、認証済みCodex CLIの既定モデルを使用します。特定モデルを固定しないため、アカウントで利用可能なCodex CLI設定に追従します。

Setupが未完了でもFake Backendによる動作確認はできます。Fakeは学習用途の代替modelではなく、workflowの再現試験用です。managed runtimeにはCLIP画像embedding modelも含まれ、WD14と合わせて大きなbatchでGPUメモリが足りない場合は自動的にbatchを半減して再試行します。

## 3. New Project

Project Editorで以下を指定します。

1. LoRA Name: 最終safetensorsの名前。Windowsで使えない記号は入力できません。名前を入力して`Add Project`を押すと、`Project\<名前>\`へ`input-Image`、`output-model`、`base-model`を含む必要なdirectory、`project.yaml`、dataset manifest、SQLiteを作成します。この時点では学習は始まりません。
2. Preset: 人物・characterの同一性を学ぶ場合はCharacter、絵柄を学ぶ場合はStyle。
3. Caption / Tag refinement: 安全なCodex提案を自動適用するか、学習前に全画像の差分を確認するかを選びます。Runtime Codex refinementを実行すると、採用された全画像のmetadataを除いた最大辺2048 pxの縮小コピーが、最大8枚ずつOpenAIへ送られます。Rawや元画像、元ファイル名、プロジェクトpathは送られず、一時JPEGは各呼び出し後に削除されます。画像準備またはCodex refinementの失敗時はTraining前に復旧可能な停止となるため、復旧後にResumeしてください。Dataset以外のCodex reviewは既存のフォールバック方針を維持します。画像入力の`--image`は[Codex CLI reference](https://developers.openai.com/codex/cli/reference/)を参照してください。
4. Trigger Word: 手入力するか、Runtime Codexが作る3〜5候補から選んで編集します。`1girl`など一般的なDanbooru tagと衝突する値は使用できません。確定値は全captionの先頭、sampling、metadata、成果物へ同じ値で反映されます。
5. Base Model: SDXL / Illustrious互換`.safetensors`。headerを安全に検査します。
6. Images / Folders: fileを複数選択するかfolderを選択。日本語、空白、Unicodeの名前を扱えます。
7. GPU Pool: 使用を許可するGPU UUIDだけをcheck。選択していないGPUは使いません。
8. Output: 完成folderの親directory。

開発版の`Project`はrepository直下、完成版は`LoRA Factory.exe`と同じdirectoryにあります。`Add Project`は同名projectに対して安全に再実行でき、既存fileを上書きしません。残りの項目を入力して`Create LoRA`を押すと、そのprojectへ設定を保存して学習を開始します。

元画像は変更されません。LoRA Factoryはhashを確認してproject内のimmutable Raw Storeへcopyし、その後の加工はworking copyにだけ行います。

## 4. CharacterとStyle

Characterは`trigger + stable class token + 画像ごとの可変tag`を基本にします。服装、表情、pose、背景などを全画像共通のidentity属性として固定しません。`1girl`等のclass tokenは無条件削除しません。

Styleは人物名・作品名・style名そのものをcaptionから除き、subject、構図、背景、lighting等のsemantic contentを残します。同じcharacterや同じ構図に偏りすぎる場合はquality gateが警告または停止します。

defaultではcrop、upscale、flip、color augmentation、random cropはoffです。aspect ratio bucketはon、bucket upscaleはoffです。

## 5. Dataset Review

Review画面ではthumbnail、元file名、解像度、判定理由、raw tags、final caption、採用状態を確認します。低解像度、極端なaspect、blank、重複、watermark候補等には理由が表示されます。

1枚に複数の意味がある場合は、`Accepted, Warning, Duplicate, Validation`のようにカテゴリを重ねて表示します。カテゴリfilterも重複して数えるため、WarningやDuplicateをValidationへ割り当てた場合でも見落としません。Validation割当はseed固定のTraining Plannerと同じsplitをDataset Review時点で計算し、後段で一致を再検証します。

Codex候補の選択または差分確認が必要なrunは、Training前に正常な`AWAITING_REVIEW`となります。元tag、検証済みの追加・削除tag、提案tag、triggerなしcaption下書き、提案caption、理由、confidenceを全採用画像について確認し、画像ごとに採用・却下・編集するか、一括採用します。未知tag、禁止tag、重複、上限超過、Trigger Word・class token・semantic coverageを満たさないcaptionは保存前に行単位で表示され、修正するまでContinueできません。「Approve and continue training」は検証済みの判断だけをatomic保存し、同じrunを再開します。アプリを閉じても確認状態は失われません。上流データが変わった場合は古い承認を拒否します。

手入力Trigger Wordと自動適用の組合せだけは停止せず最後まで進みます。Codex候補＋自動適用は候補選択だけ、手入力＋確認は画像差分だけ、Codex候補＋確認は両方の確定を待ちます。

各行には、最終training resolutionと「upscaleしない」規則から求めた具体的なbucketも表示します。短辺が64 px未満へ丸められる極端なaspect比は`Unavailable (<64 px)`となり、例外でrun全体を止めません。

Include/Excludeとcaption修正は次のrun snapshotへ保存されます。自動Rejectを手動Includeした画像は、元の判定理由を失わず`Accepted + Warning`として再評価され、tagging、caption、quality gate、Dataset TOMLの採用枚数を同じ値に保ちます。ただしdecode不能・animated imageなど、安全にworking imageを作れない入力はIncludeしてもtrainingへ戻せません。

Characterは最低8枚、Styleは最低16枚のaccepted画像が必要です。少数datasetは警告付きで進められる場合があります。validation splitは十分な枚数があるときだけ有効になります。

## 6. Training、cancel、resume

Start後はstage、epoch/step、loss、GPU、checkpoint、sample progressを表示します。Cancelは現在の処理へ安全に通知し、完成済みartifactとresume stateを残します。Recent Projectsから中断projectを選びResumeできます。

外部processを強制終了した場合も、次回起動時にRUNNING stateをrecoverableとして扱います。optimizer stateがないweight-only restartは同じresumeではなく別attemptです。disk不足で停止した場合は空き容量を確保してから同じprojectをResumeすると、保存済みstateとattempt履歴を保持したまま再開します。

## 7. Completion

Completion画面には推奨model、weight、preview、comparison、alternatives、出力pathが表示されます。最終folderには次が含まれます。

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

SettingsへA1111、Forge、ComfyUIのrootまたはLoRA folderを登録するとCopyできます。同名fileは上書きせずversioned nameを使い、copy後hashを確認します。

## 8. 困ったとき

- Setupが赤い: `.\scripts\run_dev.ps1 -Console`で起動し、表示された具体的なcheckを確認します。
- modelが拒否される: SDXLのUNetと2つ目のtext encoderを含むsafetensorsか確認します。CLIでは`uv run lora-factory inspect-model <path>`で診断できます。
- GPUが見つからない: NVIDIA driverと`nvidia-smi`を確認し、再起動後にSetupを再実行します。
- runtimeがNOT READY: Setupの「Managed Training Runtime」で`Repair`を押し、PyTorch/ONNX/sd-scripts各行の具体的な結果を確認します。開発者向けの同等診断は`uv run lora-factory doctor --deep --strict`です。
- disk不足: project outputとmanaged runtimeを合わせ、十分な空き容量があるdriveを選びます。
- Codexが利用できない: Dataset画像refinementはfallbackせず、warningとauditを残してrecoverable failureとなります。Codex復旧後にResumeしてください。Dataset以外のreviewはfallback許可時に決定論的reviewへ移ります。認証fileやAPI keyをprojectへcopyしません。
- Codex候補が3件未満: 有効なasset応答が得られていれば失敗にはせず、正常な`AWAITING_REVIEW`で停止してTrigger Wordの手入力を求めます。入力を検証・保存した後、同じrunをResumeします。

障害解析用CLIの詳細は[developer guide](developer-guide.md)にあります。
