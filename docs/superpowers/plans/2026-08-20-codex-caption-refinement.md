# Runtime Codex Caption Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GUIパイプライン内でRuntime Codexが全採用画像のtag/captionを安全に調整し、Trigger Wordはユーザー入力を優先、空欄だけCodexで補完して学習・評価・packagingまで完結できるようにする。

**Architecture:** WD14結果を不変の証拠として残し、triggerなしcaption下書き、Codex提案、Factory検証済みeffective tags、確定captionを別々のstage artifactとして扱う。確認が必要なrunは`AWAITING_REVIEW`で正常終了し、fingerprint付き承認を保存した同じrunだけを下流へresumeする。

**Tech Stack:** Python 3.12、Pydantic v2、SQLAlchemy/SQLite、PySide6、pytest/pytest-qt、Ruff、mypy

**Spec:** `docs/superpowers/specs/2026-08-20-codex-caption-refinement-design.md`

## Global Constraints

- Gitのstage、commit、push、branch操作を行わない。
- Raw画像と全`dataset/raw/`を変更しない。captionは派生datasetだけへmaterializeする。
- Runtime Codexは専用scratch Git repositoryのread-only sandboxで動かし、画像、Raw path、project path、GPU UUID、認証情報、学習権限を渡さない。
- Pydanticを設定・process境界に使い、Codex応答はFactory側で再検証する。
- PySide6 main threadで重い処理を行わず、GUIはApplication Servicesだけを呼ぶ。
- stageは永続化、冪等性、cancel、安全なresumeを維持する。
- `shell=True`、`os.system`、未検証command stringを使わない。
- Python 3.12とproject-local uv環境を使用する。

## Purpose and acceptance

Project Editorで`auto/review`と`manual/codex_suggest`を保存・復元できる。全採用assetに対するCodex提案はFactory側で再検証し、Trigger Wordは入力済み値を保持、空欄では検証済みCodex先頭候補を自動採用する。安全な候補不足またはreview指定のrunはTraining前に`AWAITING_REVIEW`となり、再起動後にも復元・編集・承認でき、stale承認を拒否して同一runを再開する。自動モードでTrigger Wordを解決できる場合は、学習・評価・packagingを確認停止なしで完走する。

## Progress

- [x] 2026-08-20: 承認済み設計、既存pipeline、設定、Codex、GUI、永続化境界を確認した。
- [x] 2026-08-20: 設定とTrigger Word validatorを追加し、focused unit/GUI 380件が成功した。
- [x] 2026-08-20: triggerなしdraft、effective tag検証、chunk処理を追加した。
- [x] 2026-08-20: Codex dataset refinement契約とfallbackを追加した。
- [x] 2026-08-20: stage、確認待ち、承認fingerprint、resumeを追加した。
- [x] 2026-08-20: GUI設定・差分確認・承認再開を追加した。
- [x] 2026-08-20: metadata、docs、全品質gateを完了した。
- [x] 2026-08-24: ユーザー入力優先・空欄Codex補完へ更新し、TDD、GUI、文書、全品質gate、Character/Style Fake E2Eを完了した。

## Surprises & Discoveries

- 2026-08-20: 現行`_execute()`は全stageを一括実行し、正常な途中停止を表現しない。`AWAITING_REVIEW`は例外ではなく、前半stage実行後の明示的returnとして組み込む必要がある。
- 2026-08-20: `ProjectConfig.schema_version`は1のままでextra forbidだが、default付きfield追加なら旧projectを互換読込できる。
- 2026-08-24: `trigger_word_mode`の既定を`codex_suggest`へ変えても、既存の非空Trigger Tokenは解決時に最優先されるため値と実行結果を維持できる。空欄だけが新たにCodex委任となる。
- 2026-08-20: `CaptionRow`は最終caption向けであり、draft/proposal/approvalはstage artifactとrun directory JSONを正本にするとmigrationなしで既存DBを維持できる。

## Decision Log

- 2026-08-20: 独立CLIは追加せずGUI/Application Service経路だけを変更する。
- 2026-08-20: 元WD14 tagを上書きせず、Codex結果は`effective_tags`として保持する。
- 2026-08-20: 確認情報はrun directoryのatomic JSONとstage artifactへ保存し、承認fingerprint一致時だけresumeする。
- 2026-08-20: Git操作禁止のため、writing-plans標準のcommit stepはすべて省略する。
- 2026-08-24: Trigger Word解決順序を「ユーザー入力 → Factory検証済みCodex先頭候補 → `AWAITING_REVIEW`」に固定する。review modeでも解決済み値を初期表示し、ユーザーは変更できる。

## Architecture and milestones

1. 設定境界: enumと条件付きTrigger Word validationを確立する。
2. Domain境界: draft生成、proposal再検証、候補検証、caption確定を純粋関数として作る。
3. Codex境界: strict schemaと128 KiB以下の決定論的chunk callを作る。
4. Orchestration境界: refinementまでと承認後を分け、durable wait/resumeを実現する。
5. GUI境界: 設定、差分、候補、個別/一括判断をApplication Service API経由で操作する。
6. Artifact境界: training、sampling、metadata、packageに同じ確定Trigger Wordを伝播する。

### Task 1: Configuration and Trigger Word contract

**Files:**
- Modify: `src/lora_factory/config/models.py`
- Modify: `src/lora_factory/gui/project_editor.py`
- Test: `tests/unit/test_foundation.py`
- Test: `tests/gui/test_gui_workflow.py`

**Interfaces:**
- Produces: `CodexRefinementMode(auto, review)`, `TriggerWordMode(manual, codex_suggest)`, `validate_trigger_word(value: str) -> str`, default-compatible `ProjectConfig` fields.

- [x] Add failing model tests proving old payload defaults to manual/auto, manual rejects empty, and codex_suggest accepts empty pending value.
- [x] Run the focused foundation test and confirm the new enum/validation tests fail at import before implementation.
- [x] Implement the enums, shared validator, and an after-model validator:

```python
class CodexRefinementMode(StrEnum):
    AUTO = "auto"
    REVIEW = "review"


class TriggerWordMode(StrEnum):
    MANUAL = "manual"
    CODEX_SUGGEST = "codex_suggest"


@model_validator(mode="after")
def validate_trigger_requirement(self) -> ProjectConfig:
    if self.trigger_word_mode is TriggerWordMode.MANUAL:
        validate_trigger_word(self.trigger_token)
    return self
```

- [x] Add two combo boxes to Project Editor, conditionally require the Trigger Word, include values in `ProjectConfig`, and restore them in `load_project()`.
- [x] Run focused unit and GUI tests and confirm 380 passed.

### Task 2: Caption refinement domain

**Files:**
- Create: `src/lora_factory/caption/refinement.py`
- Modify: `src/lora_factory/caption/character_policy.py`
- Modify: `src/lora_factory/caption/style_policy.py`
- Modify: `src/lora_factory/caption/__init__.py`
- Create: `tests/unit/test_caption_refinement.py`

**Interfaces:**
- Consumes: `PresetKind`, `TagOntology`, existing Character/Style policy configs.
- Produces: `CaptionDraftBatch`, `RefinementAsset`, `RefinementDecision`, `TriggerCandidate`, `build_caption_drafts(...)`, `validate_refinement_response(...)`, `finalize_captions(...)`, `chunk_refinement_assets(...)`.

- [x] Write failing tests for trigger-free drafts, stable Character class/invariants, Style forbidden categories, input-subset enforcement, per-asset fallback, candidate normalization/collision rejection, final Trigger-first captions, and stable chunks whose serialized input is at most 128 KiB.
- [x] Run the focused caption refinement test and confirm collection/API failures.
- [x] Reuse the existing Character/Style policies through a trigger-free draft adapter and preserve their public behavior.
- [x] Implement frozen Pydantic domain models and deterministic validators. Chunk by sorted `asset_id`, measuring canonical UTF-8 JSON bytes before appending an asset.
- [x] Run caption policy and refinement tests and confirm pass.

### Task 3: Runtime Codex structured contract

**Files:**
- Modify: `src/lora_factory/codex/schemas.py`
- Modify: `src/lora_factory/codex/fallback.py`
- Modify: `src/lora_factory/codex/prompts.py`
- Modify: `src/lora_factory/codex/gateway.py`
- Modify: `src/lora_factory/codex/scratch_repo.py`
- Test: `tests/unit/test_codex_static_schemas.py`
- Test: `tests/unit/test_codex_gateway.py`

**Interfaces:**
- Consumes: sanitized chunk dictionaries only.
- Produces: `CodexTaskType.DATASET_REFINEMENT`, `DatasetRefinementResponse`, `AssetRefinementProposal`, `TriggerWordCandidate` and matching static JSON schema.

- [x] Add failing schema parity tests for strict required fields, unknown fields, confidence bounds, decision enum, and candidate limits.
- [x] Run focused Codex schema/gateway tests and confirm failure.
- [x] Add strict Pydantic response models and gateway mapping; ensure the prompt explicitly forbids adding tags absent from the input and seeing images/paths.
- [x] Implement deterministic fallback that returns every input asset as `keep` with original effective tags and no fabricated candidate.
- [x] Run focused Codex tests and confirm pass.

### Task 4: Pipeline stages and durable review state

**Files:**
- Modify: `src/lora_factory/core/stage.py`
- Modify: `src/lora_factory/application/service.py`
- Modify: `src/lora_factory/application/artifact_store.py`
- Modify: `src/lora_factory/storage/orm.py`
- Create: `tests/integration/test_codex_refinement_pipeline.py`

**Interfaces:**
- Produces: `CAPTION_DRAFTING`, `CODEX_REFINEMENT`, `CAPTIONING`, `RunStatus.AWAITING_REVIEW`; review artifact `dataset/refinement-review.json`; `get_refinement_review(project_id)`, `approve_refinement(project_id, decision)`.

- [x] Write integration tests for the four behavior-matrix combinations and assert no `TRAINING` row exists before approval.
- [x] Run the new integration test and confirm missing stage/status/API failures.
- [x] Replace old caption stage input with draft → refinement → final caption. Call Codex once per deterministic chunk, revalidate each response, retain valid chunks by stage input hash, and record per-asset rejection reasons.
- [x] Split `_execute()` at the review barrier and return normal durable `AWAITING_REVIEW` before Training.
- [x] Implement approval models and an upstream fingerprint, reject stale approval, and retain the same `run_id` on resume.
- [x] Run the new integration tests and existing application resume tests.

### Task 5: Review persistence and restart recovery

**Files:**
- Create: `src/lora_factory/application/refinement_review.py`
- Modify: `src/lora_factory/application/service.py`
- Modify: `src/lora_factory/gui/contracts.py`
- Test: `tests/integration/test_codex_refinement_pipeline.py`

**Interfaces:**
- Produces: `RefinementReviewState`, `RefinementApproval`, atomic load/save functions, controller methods `refinement_review()` and `submit_refinement_review()`.

- [x] Add tests for restart reload, accept/reject/edit, bulk accept, Trigger validation, stale approval, and Trigger-only downstream invalidation.
- [x] Implement Pydantic boundary models; require every accepted asset exactly once and store user caption edits separately from Codex proposals.
- [x] On resume, load approval before downstream execution, inject only validated decisions, and reuse matching upstream stages/chunks.
- [x] Run restart/resume tests and confirm Training has not created an attempt at wait.

### Task 6: GUI review and resume workflow

**Files:**
- Modify: `src/lora_factory/gui/contracts.py`
- Modify: `src/lora_factory/gui/dataset_review.py`
- Modify: `src/lora_factory/gui/main_window.py`
- Modify: `src/lora_factory/gui/workers.py`
- Test: `tests/gui/test_gui_workflow.py`

**Interfaces:**
- Consumes: controller review load/submit APIs and `AWAITING_REVIEW` result.
- Produces: all-image before/after review, candidate selection/edit, per-item accept/reject/edit, bulk accept, approve-and-resume action.

- [x] Add pytest-qt tests for mode persistence, candidate validation, complete diff rendering, decisions, disabled continue state, wait state, and resume.
- [x] Extend `DatasetItemView` with immutable original/proposed fields, reason, confidence, and decision without exposing Raw paths.
- [x] Render all accepted images and add explicit buttons/actions; resume only after atomic approval succeeds.
- [x] Include `awaiting_review` in Recent Projects running/attention filters and route activation to Dataset Review.
- [x] Run GUI tests and confirm main-thread responsiveness and correct button states.

### Task 7: Metadata, documentation, and end-to-end consistency

**Files:**
- Modify: `src/lora_factory/application/service.py`
- Modify: `src/lora_factory/packaging/metadata.py`
- Modify: `docs/user-guide-ja.md`
- Modify: `docs/architecture.md`
- Modify: `tests/integration/test_application_fake_e2e.py`
- Modify: `tests/e2e/test_cli_fake_e2e.py`

**Interfaces:**
- Consumes: final Trigger Word, refinement audit, approval fingerprint.
- Produces: consistent caption files, dataset TOML, sample prompts, model metadata, reproducibility manifest, packaged README metadata.

- [x] Add integration assertions that final Trigger Word is identical across captions, samples, metadata, and package, and that Raw hashes are unchanged.
- [x] Add refinement mode, Trigger mode, final Trigger Word, fallback state, chunk audit hashes, and approval fingerprint to path-free reproducibility metadata.
- [x] Document the four GUI modes, wait/resume semantics, fallback, and the fact that Codex never sees images or paths.
- [x] Run Character and Style Fake E2E tests and confirm READY plus Raw hash equality.

### Task 8: Full validation and final review

**Files:**
- Modify: this plan's Progress, Validation, and Outcomes sections with actual results.
- Modify: common Obsidian LoRA Factory work log with safe Japanese summary.

- [x] Run Ruff format check using the project-local executable because `uv run` hit the recorded Windows cache/trampoline failures.
- [x] Run Ruff lint using the project-local executable.
- [x] Run mypy over `src` using the project-local Python.
- [x] Run focused unit, integration, and GUI suites, then full pytest.
- [x] Run pytest branch coverage and exceed the configured 80% gate.
- [x] Run Character Fake E2E with the project-local CLI module.
- [x] Run Style Fake E2E with the project-local CLI module.
- [x] Verify Codex CLI status; record that `live_codex` was not run because CLI 0.147.0 reported `Not logged in`.
- [x] Inspect `git diff --check`, `git status --short`, final diff, conflict markers, secret patterns, and temporary artifacts without staging or committing.
- [x] Verify the Obsidian log exists, title/frontmatter/body/Wikilinks are valid, and no duplicate terminal-specific note was created.

## Validation

Expected observations are: focused tests pass after each milestone; review-required runs return `AWAITING_REVIEW` before any Training stage; approved same-run resume reaches `READY`; Trigger Word matches every downstream consumer; Raw hashes before/after match; full quality gates pass. Actual outcomes will be appended with timestamps as commands complete.

- 2026-08-20: `uv run` failed before collection with the known default-cache OS error 183; repo-local uv cache then hit the known trampoline canonicalization error. Direct project interpreter execution succeeded: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_foundation.py tests/gui/test_gui_workflow.py -q` → 380 passed.
- 2026-08-20: Final Ruff format check: 201 files already formatted. Ruff lint: all checks passed. mypy: 143 source files, no issues.
- 2026-08-20: Full pytest: 626 passed, 5 skipped. Branch coverage: 82.81% (configured gate 80%).
- 2026-08-20: Character and Style Fake E2E both reached `READY`; source hashes remained unchanged. Character observed 21 stages including `CAPTION_DRAFTING` and `CODEX_REFINEMENT`; Style reported the same stage graph.
- 2026-08-20: Codex CLI 0.147.0 was installed but `login status` returned `Not logged in`; the opt-in `live_codex` test was not run. Four other live tests were skipped because the pinned managed runtime is not installed.

- 2026-08-24: TDD REDは、入力済み`codex_suggest`が`AWAITING_REVIEW`になること、空欄＋安全な候補が同じく待機すること、ProjectConfig/GUIがmanual既定であることを各公開境界で再現した。
- 2026-08-24: focused unit/integration/GUIは440 passed。Ruff formatは189 files、Ruff lintは指摘0、mypyは146 source filesで成功した。
- 2026-08-24: 全pytestは700 passed、4 skipped、branch coverage 83.29%で80% gateを通過した。skipは明示opt-in実Codex 2件とlive base model未設定2件。
- 2026-08-24: Character/Style Fake E2Eは各18画像、21 stages、165 eventsで`READY`。両方でsource hash不変を確認した。
## Outcomes & Retrospective

Runtime Codex refinement, Trigger Word modes, durable review/resume, GUI approval, audit metadata, static schema, documentation, and regression coverage are implemented. All non-live quality gates pass. The design spec, ExecPlan, implementation, and tests remain unstaged and uncommitted on branch `v0.3` per user instruction.
### 2026-08-24 User-or-Codex Trigger policy extension

ユーザーが入力したTrigger WordはCodex候補より優先して保持し、空欄の場合だけ検証済み候補の先頭を自動採用する。候補不足では従来どおりTraining前にdurable waitし、review modeでは解決済み値を編集可能な初期値として表示する。既存のmanual mode、Factory validation、same-run resume、Raw不変性を維持した。
