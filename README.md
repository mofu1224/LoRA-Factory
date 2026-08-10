# LoRA Factory

LoRA Factoryは、手元の画像からSDXL / Illustrious互換LoRAを作るWindows 11向けPySide6デスクトップアプリです。画像の取り込み、品質確認、WD14タグ付け、Character / Style別caption、学習計画、学習、checkpoint比較、最終モデルの選択と梱包までをGUIから進めます。

ChatGPTで利用できるCodex CLIを認証しておけば、API keyをrepositoryへ保存せず、datasetと学習結果の構造化レビューを利用できます。Codexが一時的に利用できない場合も、許可設定に応じて決定論的fallbackでworkflowを継続できます。

元画像と各projectの`dataset/raw/`は変更しません。取り込みはSHA-256検証付きcopyで行い、crop・resize・caption等は派生領域だけへ保存します。

複数GPUを選ぶとWD14 taggingと参照／生成CLIP embeddingをUUID単位で分割し、samplingも選択pool内でrotationします。標準学習jobは品質と互換性を優先して1 GPUで実行し、開始前のselected-GPU batch probeで実測free VRAMを確認します。

## 必要なもの

- Windows 11 x64
- NVIDIA CUDA GPUと最新の対応driver（実学習時）
- 15 GiB以上のmanaged runtime領域に加え、base model・project・出力を保存できる空き容量
- SDXL / Illustrious互換の`.safetensors` base model
- Codex CLIを利用できるChatGPTアカウント

## 一般ユーザー向けの最短手順

1. [Codex CLI公式ドキュメント](https://learn.chatgpt.com/docs/codex/cli)に従ってCodex CLIをinstallします。
2. PowerShellで`codex`を一度実行し、初回画面で`Sign in with ChatGPT`を選んで認証します。
3. GitHub Releasesから`LoRA-Factory-*-windows-x64.zip`と`.sha256`をdownloadし、hashを確認してから書き込み可能なfolderへ展開します。
4. `LoRA Factory.exe`を起動し、Setup画面のcheckを実行します。
5. `Managed Training Runtime`の`Repair`を押して、download・install・GPU検査が完了するまで待ちます。
6. `New Project`でbase model、画像、使用GPU、出力先を指定し、Dataset Reviewを確認して学習を開始します。

Windows ZIPにはアプリ本体とlicenseを含みますが、CUDA/PyTorch/sd-scripts/WD14/CLIP modelは含みません。初回Setupがpin済みmanaged runtimeとして別途installします。base modelと学習画像はユーザー自身で用意してください。

## ソースから起動する開発者向け手順

Python 3.12、[uv](https://docs.astral.sh/uv/)、Gitを用意し、PowerShellでrepository直下から次を実行します。

```powershell
.\scripts\bootstrap.ps1
.\scripts\run_dev.ps1
```

`run_dev.ps1`は通常、コンソールを表示せずGUIを起動します。エラーを画面で確認したい場合だけ次を使います。

```powershell
.\scripts\run_dev.ps1 -Console
```

初回はGUIのSetup画面で環境を確認します。Real Backendを使う前に「Managed Training Runtime」の`Repair`を押してください。約15 GiB以上を使う分離runtimeの構築と、実GPU上のPyTorch/bf16・ONNX CUDA・sd-scripts検査をbackgroundで完了します。

Runtime Codexレビューは、認証済みCodex CLIの既定モデルを使います。特定モデルを固定しないため、一般のChatGPTアカウントで利用可能な最新のCodex CLI設定に追従します。

`New Project`でLoRA名を入力して`Add Project`を押すと、開発版ではrepository直下、完成版ではEXEと同じdirectoryの`Project\<LoRA名>\`へ、`input-Image`、`output-model`、`base-model`、dataset/config/runs/final、manifest、SQLiteを含むproject構造をすぐ作成します。この操作だけでは学習を開始しません。残りの項目を設定して`Create LoRA`を押すと、同じdraft projectを本設定へ昇格してpipelineを開始します。

CUDA/PyTorch/sd-scripts/WD14/CLIP modelはGUI executableへ埋め込みません。`%LOCALAPPDATA%\LoRAFactory\runtimes\`以下のversioned runtimeとして管理します。

## データとプライバシー

- 元画像と`dataset/raw/`は読み取り専用の入力として扱い、crop・resize・caption等は派生領域だけへ保存します。
- Runtime Codexは専用scratch repository内のallowlist済み構造化metadataだけを読み、元画像・model weight・認証file・API keyは渡しません。
- 完成LoRAに添付するJSON/YAML/TXTから、絶対path、元画像名、内部command、物理GPU UUIDを除去します。
- GitHubへIssueを作る場合も、画像、model、token、個人pathを添付しないでください。

## 再現可能なFake E2E

GPUや実モデルを使わず、validな小型SDXL safetensorsと18枚の多様なPNGを新規生成し、GUIと同じApplicationController pipelineを最後まで通します。

```powershell
uv run --frozen lora-factory fake-e2e
```

既存directoryは上書きせず、`.artifacts\fake-e2e\run-<timestamp>-<id>\`へ結果と`fake-e2e-report.json`を保存します。fixture元画像の実行前後SHA-256一致も合格条件です。

## 開発用CLI

```powershell
uv run --frozen lora-factory doctor
uv run --frozen lora-factory doctor --deep --gpu-uuid GPU-...
uv run --frozen lora-factory inspect-model "D:\Models\base.safetensors"
uv run --frozen lora-factory fake-e2e --preset character
uv run --frozen lora-factory fake-e2e `
  --gpu-uuid GPU-aaaaaaaa-1111-2222-3333-444444444444 `
  --gpu-uuid GPU-bbbbbbbb-1111-2222-3333-444444444444
```

Fake E2Eの`--gpu-uuid`は繰り返し指定でき、物理GPUを検出・使用せずにtaggingと
evaluationのsharding/rotationを確認します。未指定時は固定のFake GPU 1台です。CLIは
開発・テスト・障害解析専用で、通常ユーザーのworkflowはGUIで完結します。

## Windows build

```powershell
.\scripts\build_windows.ps1
```

PyInstaller one-dir成果物を`dist\windows-<timestamp>\LoRA Factory\`へ作ります。既存buildを上書きしません。詳細は[developer guide](docs/developer-guide.md)を参照してください。

## Backend pin

`backend-manifest.json`が管理runtimeのsource of truthで、`runtime-lock.txt`をSHA-256で
固定します。sd-scripts checkout内の可変な`requirements.txt`を直接installしません。

- Python 3.12.13
- unmodified `kohya-ss/sd-scripts` `v0.11.1`, commit `6721028c79ee85a78b3a06dfd8954dae310a1cce`
- PyTorch 2.13.0 / torchvision 0.28.0, official cu130 wheels
- ONNX Runtime GPU 1.28.0
- WD EVA02 Large Tagger v3 `v1.0`, revision `c5303bb7139430db980e4c680a778fe79d72b541`
- OpenAI CLIP ViT-L/14, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`
- Windows managed runtime 57 packagesのexact-version lock

host baselineはCUDA Toolkit 13.1です。managed wheelは公開済みの最も近いCUDA 13 runtimeを使い、`sm_120`、CUDA tensor、bf16、ONNX CUDA provider、sd-scripts importをGUI SetupまたはRuntime Doctorで実測するまでhealthyとは扱いません。

WD14とCLIP helperはOOM時にbatchを半減し、失敗batchのGPU tensorを解放してから再試行します。WD14の成功batch sizeはGPU UUID／model revision別にcacheします。

## Documentation

- [日本語ユーザーガイド](docs/user-guide-ja.md)
- [Architecture](docs/architecture.md)
- [Developer guide](docs/developer-guide.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

LoRA Factory本体はMIT Licenseです。Windows buildにはCPythonと各bundled distributionの
license/noticeを収集した`licenses\`を同梱します。base modelや生成物の利用条件は各model
licenseを確認してください。
