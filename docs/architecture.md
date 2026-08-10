# LoRA Factory architecture

## Boundaries

依存方向は`gui -> application -> core/domain`です。GUI widgetはdatabase、trainer command、raw imageを直接操作せず、Application Serviceだけを呼びます。ApplicationControllerは同期headless APIとして定義され、Qtはworker threadから呼び出してimmutableなevent payloadをmain threadへ返します。

主要境界は次の通りです。

- `gui/`: setup、project editor、dataset review、progress、completion、settings。重処理をしない。
- `application/`: project単位のworkflow orchestration、backend選択、event変換。
- `core/`: stage state machine、fingerprint、cancellation、resume。
- `project/`, `dataset/`, `caption/`: immutable importと派生dataset構築。
- `gpu/`, `runtime/`: UUID-first schedulingと分離backend runtime。
- `training/`, `sampling/`, `evaluation/`:同じprotocolを実装するReal/Fake adapters。
- `codex/`: sanitized structured reviewだけを行うread-only gateway。
- `packaging/`: final、alternatives、preview、comparison、manifestと安全なdestination copy。
- `storage/`: project-local SQLite/Alembic stateとaudit event。

## Durable project model

各projectは独立したdirectoryとSQLite databaseを持ちます。代表的なlayoutは次です。

```text
project.yaml
state.sqlite3
dataset/
  raw/          immutable content-hash copies
  working/      normalized PNGs
  captions/     derived captions
  training/     accepted training copies and captions
validation/
configs/        generated sd-scripts TOML
runs/<run-id>/
  logs/
  checkpoints/
  samples/
  evaluation/
  codex/
completion.json
```

`dataset/raw/`のfile名はcontent identityに基づき、元のpath/file名はmanifestへ記録します。source/rawにはcaption、crop、resize、rename、delete、hardlinkを行いません。Pipeline前後のhash mismatchは失敗です。

## Pipeline and recovery

順序はimport、normalize、analyze、deduplicate、tag、caption、dataset review、plan、Codex pretrain review、preflight、train、three-pass sampling、evaluation、Codex final review、select、package、readyです。

各stageはversion、input fingerprint、backend version、status、outputを永続化します。同じfingerprintで完成済みのstageは再利用でき、RUNNINGのまま終了したprocessは起動時にrecoverable stateへ戻します。cancelはcooperative tokenから外部processのterminate/killへ伝播し、resume可能なoptimizer stateとcheckpointを保持します。

学習attemptはdirectoryを分離し、model/dataset/plan/seed/backend/runtime fingerprintが一致するoptimizer stateだけをresumeします。OOM/NaNは最大3 attemptの決定論的ladder、disk fullは外部で容量を確保した後に同じrunをResumeできるrecoverable stateとして永続化します。

## GPU isolation

public identityはNVIDIA GPU UUIDです。実行直前に最新の`nvidia-smi` snapshotからUUIDをphysical indexへ解決し、選択pool外のdeviceは拒否します。子processは割り当てた1台だけを`CUDA_VISIBLE_DEVICES`で可視化し、その中のlogical `cuda:0`を使います。indexの並び替えを永続IDとして扱いません。

WD14 tagging、reference/generated CLIP embeddingは選択poolへ決定論的にshardし、独立leaseで並列実行して元の入力順へmergeします。sampling passはpool内でrotationし、標準training jobは1 GPUだけをleaseします。各leaseはSQLite heartbeatを持ち、crash後はstale leaseを回収します。並列GPU inventory更新はatomic upsertで競合を避けます。

WD14 helperはOOM時にbatchを半減し、成功batch sizeをGPU UUIDとmodel revisionごとにcacheします。CLIP helperも失敗batchのCUDA tensorとtraceback参照を解放してから半減再試行します。どちらもselected UUIDを再解決したchildのlogical `cuda:0`以外を使用しません。

## Dataset review and planning

Review itemはAccepted、Warning、Rejected、Duplicate、Validationを重複可能なcategoryとして保持します。include/excludeとcaption overrideは次のimmutable run snapshotにだけ取り込まれます。手動includeされた品質Rejectは理由を失わずWarningへ格上げし、tag、caption、quality gate、TOML、manifestで同じaccepted集合を使います。

表示bucketは文字列placeholderではなく、pin済みsd-scripts v0.11.1のno-upscale `BucketManager.select_bucket`と同じ64px step計算です。Dataset Reviewで最終Planningと同じresolutionを先行解決し、Planningでparityを再検証します。極端なaspectで有効な短辺bucketが作れない画像は例外にせず`Unavailable (<64 px)`としてReviewへ残します。

## Managed runtime

FactoryのPython 3.12環境と、PyTorch/ONNX/sd-scripts/WD14/CLIPを含む大容量runtimeを分離します。`backend-manifest.json`にrelease、full commit、model revision、artifact size/hash、wheel profile、`runtime-lock.txt`のSHA-256をpinします。Torch/torchvisionはofficial cu130 indexからexact versionを導入し、残る57 packagesはhash検証済みexact lockを`--no-deps`で導入します。sd-scriptsはpinned checkoutを`--no-deps --editable`で登録し、upstream `requirements.txt`による再解決やsource改変を行いません。

Runtime Doctorは選択GPUだけを可視化したchild processで次を確認します。

- PyTorch version/CUDA runtimeと`sm_120` architecture
- CUDA tensor operationとbf16 operation
- ONNX Runtime CUDA Providerとdevice transfer
- pinned `sdxl_train_network` import

Training Plannerはdataset/GPUから暫定planを作った後、学習開始前にbatch probeを実行します。Real providerは計画されたGPU UUIDを直前にphysical indexへ再解決し、その1台だけを見せたmanaged Pythonで`torch.cuda.mem_get_info()`と最大32 MiBの実CUDA allocationを行います。候補batchを大きい順に判定し、保守的な学習VRAM予約と1 GiB safety marginを満たす最初の値で最終plan、effective batch、step数を再計算します。Fake providerも同じ予約式と降順選択を決定論的に実行します。probeはcheckpoint、optimizer state、学習datasetを作らず、結果・attempt・GPU lease・log pathだけをstage stateとreproducibility manifestへ残します。

## Runtime Codex boundary

Runtime Codexはsource repositoryやraw pathを受け取りません。dedicated scratch Git repositoryへsanitized JSON、schema、promptだけを配置し、`codex exec --ephemeral --sandbox read-only --json --output-schema ...`をargument arrayで起動します。outputをPydantic schemaで再検証し、変更提案はallowlist内だけを適用します。timeout/invalid output/auth failure時はpolicyに応じて明示warning付きdeterministic fallbackへ移ります。

静的schemaはRuntimeのPydantic response modelから生成したものと完全一致させます。Codex Structured Outputsのsubsetに合わせ、rootをobject、全fieldをrequired、全objectを`additionalProperties: false`とし、`default`と表示用`title`は除去します。Dataset、Caption、Training、Recovery、Finalの5 schemaをparity testで監視します。要件は[OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)を参照してください。

## Packaging and distribution

final publicationはLoRA名へ正規化したsafetensors、最大3つのalternatives、preview、comparison、README、resolved config、evaluation、training info、reproducibility manifestを含みます。Stable Diffusion install先へのcopyは既存名を上書きせずversioningし、copy後SHA-256を検証します。

Windows GUIはPyInstaller one-dirで配布します。巨大なmanaged runtimeは同梱せず、`%LOCALAPPDATA%\LoRAFactory\`へ別途構築します。manifest、runtime lock、preset、Codex schema、外部managed processが読むWD14/CLIP helperは物理data fileとしてone-dirへ含めます。WD14とCLIPのmodel重みは含めず、hash検証付きSetupでmanaged runtimeへinstallします。build時にCPythonとbundled Python distributionsのlicense/noticeを隣接`licenses/`へexportします。これによりGUI更新とbackend互換性profileを独立して管理できます。
