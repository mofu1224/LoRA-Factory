# ベータビルド前フルデバッグ ExecPlan

## Purpose and acceptance

LoRA Factoryの現在の未コミット変更を含む作業ツリーを、Windowsベータ配布物を生成する直前の状態として監査する。正式なビルドはこの計画の対象外とし、リポジトリ定義のpublication preflight、全テストとbranch coverage、GUI、Character/Style Fake E2E、PyInstaller設定と配布ライセンス境界、差分の正確性・可読性・構造・セキュリティ・性能を検証する。再現した不具合は根本原因を特定し、意味を変えない最小修正と回帰テストで解消する。raw画像、managed runtime、vendor、ユーザーデータを変更せず、commit、push、tag、release、配布物生成を行わない。

Acceptanceは、全Critical/Required findingが解消または環境依存として明示され、Ruff format/check、mypy、全pytestと80%以上のbranch coverage、GUIテスト、Character/Style Fake E2E、lock整合、`git diff --check`が成功し、Windows build script/spec/license入力を静的検査できること。実Codex、実GPU、managed runtime、ユーザー提供base model、clean Windows accountが必要な項目は、ローカルで満たせない場合に未検証と明記する。

## Progress

- [x] 2026-08-24: 共通Obsidian方針、既存LoRA Factory記録、`.agent/PLANS.md`、release checklist、preflight/build scriptを確認した。
- [x] 2026-08-24: `code-review-and-quality`と`verification-before-completion`を選定した。
- [x] 2026-08-24: 変更テストを先に読み、5軸レビューと静的危険箇所監査を完了した。
- [x] 2026-08-24: publication preflight、全テスト、coverage、GUIを実行した。
- [x] 2026-08-24: Character/Style Fake E2E、実Codex、両GPU deep doctor、build境界を検証した。
- [x] 2026-08-24: Critical/Required findingと再現不具合はなかったため、production codeの追加修正は不要と判断した。
- [x] 2026-08-24: 最終ゲート、差分、一時成果物、Obsidianログを確定した。

## Surprises & Discoveries

- 2026-08-24: `safe-debug`は具体的な障害発生後の保守的診断専用で、全体監査には不適合だったため適用を中止した。
- 2026-08-24: `code-review-and-quality`が参照する`security-checklist.md`と`performance-checklist.md`はインストール内に存在しない。スキル本文の5軸チェックとリポジトリ固有ルールで代替する。
- 2026-08-24: 作業ツリーには直前タスクのTrigger Word自動解決に関する12ファイルの未コミット変更がある。これをユーザー作業として保持し、監査対象に含める。
- 2026-08-24: project Pythonはpipを同梱しないため`python -m pip check`は実行不能だった。uv管理環境の正規手段である`uv pip check`へ切り替え、50 packagesの互換性確認に成功した。
- 2026-08-24: 狭いterminal幅で`live-smoke --help`のdual optionが文字化けして見えたが、`NO_COLOR=1`と160 columnsで再実行すると正しい`--allow-codex-fallback / --require-codex`表示だった。実装不具合ではない。
- 2026-08-24: 以前は未loginだったCodex CLIが現在はChatGPT認証済みだったため、合成JSONと96x96単色画像だけを使う2件のopt-in live testを追加実行できた。

## Decision Log

- 2026-08-24: ユーザーの`bataビルド`は文脈上`ベータビルド`と解釈する。
- 2026-08-24: 「ビルド前」のため`build_windows.ps1`による配布物生成は実行しない。代わりに同script、PyInstaller spec、license export、release workflowを静的に確認し、publication preflightまで実行する。
- 2026-08-24: 既存差分を破棄・stash・commitせず、その状態をそのまま検証する。

## Architecture and milestones

1. Tests first: Trigger Word方針のunit/integration/GUI testから意図を復元し、実装差分を5軸で照合する。
2. Static gates: lock、format、lint、type、schema、secret/conflict marker、subprocess/path境界を確認する。
3. Runtime gates: 全pytest/coverage、GUI、Character/Style Fake E2Eを一時領域を分離して実行する。
4. Packaging boundary: `packaging/lora_factory.spec`、license export、release workflow、build scriptの入力・出力・上書き拒否・VC runtime条件を確認する。
5. Fix loop: 失敗を単独再現し、原因を特定してから最小修正と回帰テストを行い、全ゲートを再実行する。

## Validation

- `git diff --check`
- `uv lock --check`
- `scripts/preflight_publication.ps1`
- `.venv\Scripts\python.exe -m ruff format --check .`
- `.venv\Scripts\python.exe -m ruff check .`
- `.venv\Scripts\python.exe -m mypy src --no-incremental`
- `.venv\Scripts\python.exe -m pytest --cov=lora_factory --cov-branch --cov-report=term`
- `.venv\Scripts\python.exe -m pytest tests/gui -q`
- `.venv\Scripts\python.exe -m lora_factory.cli fake-e2e --preset character --image-count 18 --json`
- `.venv\Scripts\python.exe -m lora_factory.cli fake-e2e --preset style --image-count 18 --json`
- PyInstaller spec、build script、release workflow、license素材の静的検査

実行結果は各milestone完了時にここへ追記する。

2026-08-24 actual:

- `scripts/preflight_publication.ps1`: exit 0。Ruffは208 files、lint指摘0、mypyは146 source filesで成功した。
- 全pytest: `700 passed, 4 skipped in 148.19s`、branch coverage `83.29%`。skipは実Codex opt-in 2件とbase model未設定のmanaged backend 2件。
- GUI単独: `24 passed`。配布ライセンス・公開除外focused test: `49 passed`。
- Character 8画像: 21 stages、133 events、`READY`。Character/Style 18画像: 各21 stages、165 events、`READY`。全3 runでsource hash不変、Codex scratch JPEG 0件。
- `LORA_FACTORY_LIVE_CODEX=1`で合成データだけを送る実Codex test: `2 passed in 96.48s`。fallbackなしのstructured reviewと画像添付refinementが成功した。
- RTX 5080 / RTX 5070 Tiを各GPU UUIDで指定したdeep doctor: 両方Required failure 0、runtime `ready=true`。RTX 5080詳細はTorch `2.13.0+cu130`、`sm_120`、bf16 tensor smoke、ONNX CUDA transfer、pinned sd-scripts importが成功した。RTX 5070 Tiも同じrequired probe setを通過した。
- `uv pip check`: 50 packages compatible。PowerShell build/preflight scriptとPyInstaller specの構文検査に成功した。
- frozen build extraからPyInstaller `6.22.0`を導入・起動でき、license exporterは120 filesとCPython LICENSE 1件を一時領域へ生成した。Windows配布物は生成していない。
- 静的監査: `shell=True`、`os.system`、危険なdeserialization、秘密値候補、競合markerは検出なし。広い例外処理はcleanup、永続化後の再送出、worker通知、process-tree所有境界として意図を確認した。

## Outcomes & Retrospective

ベータビルド前のローカル品質ゲートは完了した。現在のTrigger Word自動解決差分にCritical/Required findingや再現不具合はなく、production codeの追加修正は行っていない。非live全テスト、coverage、GUI、Character/Style Fake E2E、認証済みCodexのstructured/image境界、両GPUのmanaged runtime deep probe、frozen build extra、PyInstaller起動、license exportを実測した。

環境依存で残るのは、ユーザー提供の実SDXL/Illustrious base modelと非私的datasetを使うone-epoch Real Backend pipeline、`build_windows.ps1 -SystemVcRuntime`による実ZIP生成、clean Windows accountでの起動・Setup・Fake Backend project確認である。これらはベータビルド工程または実モデル検証工程で行う。Task中に生成したtest temp、Fake E2E run、license export、PyInstaller cacheは削除し、`.venv`は通常のfrozen環境へ戻した。commit、push、tag、releaseは行っていない。
