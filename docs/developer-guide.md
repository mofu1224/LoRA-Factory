# LoRA Factory developer guide

## Toolchain

- Windows 11
- Python `>=3.12,<3.13`
- uv
- Git
- CUDA Toolkit 13.1 host baseline

Factory dependencyはrepository-local `.venv`へだけinstallします。

```powershell
.\scripts\bootstrap.ps1
```

lockfileを更新する意図があるときだけ`uv lock`を実行してください。通常の再現実行は`--frozen`を使います。

## Launch

```powershell
# consoleなし
.\scripts\run_dev.ps1

# traceback/log確認用
.\scripts\run_dev.ps1 -Console

# 同等のforeground entry point
uv run --frozen lora-factory-gui
```

GUI main threadは重処理を行いません。headless ApplicationControllerをQt workerから呼び、GUIはevent payloadだけを表示します。

name-onlyの`Add Project`は`application_root()/Project/<name>`へdraft `project.yaml`、dataset manifest、`input-Image`、`output-model`、`base-model`を含むcanonical directory tree、初期化済みSQLiteを作ります。frozen時の`application_root()`は`sys.executable`の親、開発時はrepository rootです。full `ProjectConfig`は`Create LoRA`時に同じdraftをatomicに置き換えます。

## Quality gates

```powershell
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run pytest --cov=lora_factory --cov-branch --cov-report=term
```

Core coverageのrelease gateはbranch込み80%以上です。通常のfull suiteを最後まで実行したcoverage dataに対して判定し、focused testだけの高い数値をrelease証拠にしません。

Live testsは通常suiteからmarkerで分けます。

```powershell
uv run pytest -m live_gpu
uv run pytest -m live_codex
uv run pytest -m live_sd_scripts
```

実行していないLive testをpassingとして報告しないでください。

## CLI diagnostics

```powershell
uv run --frozen lora-factory doctor
uv run --frozen lora-factory doctor --json
uv run --frozen lora-factory inspect-model "D:\Models\sdxl.safetensors" --json
```

`doctor --strict`はerrorまたは未setupでexit code 2です。`doctor --deep`は選択GPUだけをchildへ見せ、PyTorch CUDA/bf16、architecture list、ONNX CUDA Provider、sd-scripts importを実行します。

## Managed runtime

```powershell
uv run --frozen lora-factory install-runtime
uv run --frozen lora-factory doctor --deep --strict
```

default rootは`%LOCALAPPDATA%\LoRAFactory\runtimes`です。別driveを使う場合は両commandへ同じ`--runtime-root`を渡します。installerはHTTPSからpin済みartifactを取得し、size、SHA-256またはGit blob hashを検証します。partial WD14/CLIP downloadは`.partial`から再開します。`runtime-lock.txt`はmanifest記録のSHA-256と一致しなければinstall前に停止します。Torch/torchvision以外の57 packagesはexact lockをdependency resolutionなしで導入し、pinned sd-scriptsは`--no-deps --editable`で登録します。checkoutの`requirements.txt`を直接installせず、sd-scripts checkoutも変更しないでください。

## Fake E2E

```powershell
uv run --frozen lora-factory fake-e2e
uv run --frozen lora-factory fake-e2e --preset style --image-count 18 --json
uv run --frozen lora-factory fake-e2e `
  --gpu-uuid GPU-aaaaaaaa-1111-2222-3333-444444444444 `
  --gpu-uuid GPU-bbbbbbbb-1111-2222-3333-444444444444
```

commandは毎回新規workspaceへvalidな小型SDXL safetensorsとPNG/JPEG/WebP画像を作り、`LoRAFactoryController.run_pipeline()`を呼びます。fixtureは日本語、空白、括弧、数字だけ、長いUnicode名と、1024x1024、1920x1080、1080x1920、1600x1200、2048x1365、extreme aspect warning、low-resolution warningを含みます。import、analysis、caption、plan、Fake training/checkpoints、sampling、evaluation、selection、packageをproductionと同じstage graphで実行し、指定matrixが手動rename/crop/resizeなしでDataset TOMLへ到達することをE2E testで確認します。最終model header/hash、preview/comparison、source fixture hash不変を検証し、`fake-e2e-report.json`を書きます。`--gpu-uuid`を繰り返すと物理GPU discoveryなしでsynthetic poolを構成し、WD14/embeddingのshardingと後段GPU rotationを検証できます。reportは互換用の先頭`gpu_uuid`と全`gpu_uuids`、最終reproducibility manifest pathを記録し、manifest側のGPU capabilitiesとも照合します。

## Real Backend smoke

managed runtime構築後、Character minimumを満たす8枚以上のtiny datasetで実行します。

```powershell
.\scripts\smoke_real_backend.ps1 `
  -BaseModel "D:\Models\sdxl.safetensors" `
  -InputPath "D:\Datasets\tiny-character" `
  -OutputRoot "D:\LoRA-Smoke" `
  -GpuUuid @("GPU-...5080", "GPU-...5070Ti")
```

`-GpuUuid`は1枚でも複数枚でも指定できます。複数時は各GPUを個別にdeep probeしてから、同じselected GPU poolをGUIと同じReal Backend pipelineへ渡します。CLIを直接使う場合は`--gpu-uuid GPU-A --gpu-uuid GPU-B`のようにoptionを繰り返します。未指定時は空きVRAM最大の1枚を選びます。scriptはresolution 768、rank 4、batch 1、repeats 1、epoch 1のbounded planを使い、trainingだけでなくsamplingとfinal packageまで成功して初めて合格です。runtime未構築時だけ明示的に`-InstallRuntime`を追加できます。

Planning直前には選択されたtraining GPU上で副作用のないBatch Probeを行います。候補batchを大きい順に試し、実free VRAMと32 MiBのbounded CUDA allocationを観測してからplanを確定します。probeはtrainerを起動せず、checkpointやoptimizer stateを作りません。WD14とCLIP helperはbatch OOM時に失敗tensorを解放してbatchを半減し、成功batch sizeをGPU・model revision単位で再利用します。複数GPU時はtagging、reference/generated embedding、samplingをselected pool内でshardまたはrotateし、1つのtraining attempt自体は1 GPUへleaseします。

## Windows build

```powershell
.\scripts\build_windows.ps1
```

scriptはbuild extraをlockfile通りに同期し、PyInstaller specからwindowed one-dir appを作ります。出力は`dist\windows-<timestamp>\LoRA Factory\`です。指定labelが既に存在すると上書きせず失敗します。

```powershell
.\scripts\build_windows.ps1 -BuildLabel local-test-001
```

build後に`LoRA Factory.exe`、README、LICENSE、THIRD_PARTY_NOTICES、`licenses\`とexecutable SHA-256を確認します。`licenses\`にはembedded CPythonとbundled Python distributionsの実license/noticeおよびversion metadataが出力されます。さらに`_internal\backend-manifest.json`、`_internal\runtime-lock.txt`、preset、Codex schema、`_internal\src\lora_factory\runtime_scripts\wd14_infer.py`、`clip_embed.py`が存在することを確認します。PyTorch、CUDA runtime、WD14/CLIP model重み、sd-scripts checkoutはexeへ埋め込みません。初回Setupでversioned managed runtimeを別途installします。

## Pin update procedure

1. official release/documentationとlicenseを確認する。
2. release tagだけでなくfull commit SHAを確定する。
3. candidate runtimeで`uv pip check`とlive smokeを通す。
4. `uv pip freeze`からTorch/torchvisionとeditable sd-scriptsを除いたWindows exact lockを更新する。
5. `backend-manifest.json`へlock SHA-256、package count、download size/hashを記録する。
6. 新しい`profile_id`を使い、既存managed runtimeを上書きしない。
7. fresh runtimeでRuntime Doctor、tiny real training、samplingを実GPUで再実行する。
8. exact command/outputとcompatibility resultをExecPlanへ追記する。

CUDA 13.1 host baselineとwheel runtime labelが一致するだけではcompatibleと判断しません。GPU compute capability、wheel architecture、実tensor/bf16 operationを証拠にします。

## Safety checklist

- `dataset/raw/`やsource imageを書き換えない。
- user inputをshell command stringへ連結しない。
- GPUはUUIDで選び、process起動直前にindex解決する。
- source treeからRuntime Codexを起動しない。
- dirty worktreeの無関係な変更をreset、format、stageしない。
- build/output/model copyで既存fileを上書きしない。
- GitHubへpushするのは明示依頼がある場合だけにする。
