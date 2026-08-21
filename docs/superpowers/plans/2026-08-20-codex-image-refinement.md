# Runtime Codex Image Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Runtime Codexへ全採用画像のmetadata除去・縮小済み派生copyを最大8枚ずつ直接添付し、画像から確認できる不足tagをpin済みWD14語彙内で安全に追加して、画像別の手作業なしに学習を完結できるようにする。

**Architecture:** 既存working PNGから決定論的な一時JPEGを専用Codex scratchへ生成し、sanitized JSONと対応付けて`codex exec --image`へargument arrayで渡す。Factoryは画像hash、asset ID、pin済みWD14語彙、preset別禁止category、strict schemaを再検証し、全画像をCodexが確認できないrunはTraining前にrecoverable停止する。派生画像は全終了経路で削除し、成功batchはworking画像hash・変換profile・派生画像hash・JSON hash一致時だけ再利用する。

**Tech Stack:** Python 3.12、Pillow、Pydantic v2、PySide6、Codex CLI、pytest/pytest-qt、Ruff、mypy

**Spec:** `docs/superpowers/specs/2026-08-20-codex-caption-refinement-design.md`

## Global Constraints

- Gitのstage、commit、push、branch操作を行わない。各task末のcommit stepはユーザー指示により省略する。
- Raw画像、全`dataset/raw/`、既存working画像を変更しない。Codex用派生画像はscratch内だけに作成する。
- Runtime Codex refinementを使うrunは全採用画像を常にOpenAIへ送信し、画像なしfallbackでTrainingを続行しない。
- 派生画像はsRGB RGB、metadataなし、長辺最大2048px、upscaleなし、JPEG品質95/90/85/80、最大8 MiBとする。
- 1 Codex callはasset ID安定順の最大8画像、sanitized JSONは128 KiB以下とする。
- Runtime Codexは専用scratch Git repositoryのread-only sandboxで動かし、Raw path、project path、元file名、GPU UUID、認証情報、学習権限を渡さない。
- Codex CLIは`subprocess` argument arrayで起動し、`shell=True`、`os.system`、未検証command stringを使わない。
- 新規tagはpin済みWD14 `selected_tags.csv`語彙に限定し、WD14元tag/confidenceは上書きしない。
- Pydanticを設定・process境界に使い、Codex応答はFactory側で再検証する。
- PySide6 main threadで画像変換やCodex callを行わず、GUIはApplication Servicesだけを呼ぶ。
- stageの永続化、冪等性、cancel、crash、安全なresumeを維持する。
- Python 3.12とproject-local `.venv`を使用する。`uv run`が既知のWindows cache/trampoline問題で失敗する場合はproject-local executableを使い、理由を記録する。

---

## Baseline

既存の未stage・未commit実装には、triggerなしdraft、tag-only Runtime Codex refinement、`auto/review`、`manual/codex_suggest`、`AWAITING_REVIEW`、approval fingerprint、same-run resume、GUI差分確認、Trigger Word伝播が含まれる。既存の完了記録は`docs/superpowers/plans/2026-08-20-codex-caption-refinement.md`に残し、このplanは画像入力差分だけを未完了taskとして管理する。

## File map

- Create `src/lora_factory/codex/image_attachment.py`: Codex専用画像のPydantic profile/result、決定論的変換、atomic write、cleanup。
- Create `src/lora_factory/caption/tag_vocabulary.py`: pin済みWD14 CSVとFake語彙の安全な読込・membership判定。
- Modify `src/lora_factory/caption/tagger.py`: Fake backendの既知語彙を公開定数として共有。
- Modify `src/lora_factory/caption/refinement.py`: 最大8枚batch、既知語彙の追加、個別tag拒否、追加・削除diff。
- Modify `src/lora_factory/codex/scratch_repo.py`: 添付画像pathのscratch containment検証と`ScratchCall.image_paths`。
- Modify `src/lora_factory/codex/process.py`: `--image <path>`の反復引数。
- Modify `src/lora_factory/codex/gateway.py`: 画像hash付きauditと画像必須dataset refinement call。
- Modify `src/lora_factory/codex/prompts.py`: 画像fileとasset ID対応、画像から確認できる既知語彙だけを追加する指示。
- Modify `src/lora_factory/application/service.py`: working画像対応、画像準備、batch cache、hard-stop、進捗、cleanup。
- Modify `src/lora_factory/application/refinement_review.py`: 追加・削除tagのreview境界。
- Modify `src/lora_factory/gui/dataset_review.py`: 追加・削除tag表示。
- Modify `src/lora_factory/gui/project_editor.py`: OpenAIへ派生画像を常時送信する説明。
- Modify `docs/architecture.md`, `docs/user-guide-ja.md`, `README.md`, `SECURITY.md`: 画像送信・privacy・failure契約を同期。
- Create `tests/unit/test_codex_image_attachment.py`: 画像変換・cleanup回帰試験。
- Modify `tests/unit/test_caption_refinement.py`, `tests/unit/test_codex_gateway.py`, `tests/unit/test_codex_static_schemas.py`: 語彙追加、batch、引数、schema/audit。
- Modify `tests/integration/test_application_fake_e2e.py`: 全画像準備、cleanup、hard-stop、resume、Raw/working hash不変。
- Modify `tests/gui/test_gui_workflow.py`: disclosure、追加・削除表示、進捗。
- Modify `tests/live/test_codex_live.py`: 小型sanitized画像を実Codex CLIへ添付するopt-in smoke test。

## Progress

- [x] 2026-08-20: OpenAI公式CLI referenceとローカルCodex CLI 0.147.0で複数`--image`入力を確認した。
- [x] 2026-08-20: 画像変換、最大8枚batch、既知語彙追加、hard-stop、cleanup、GUI・検証設計をユーザーが承認した。
- [x] 2026-08-20: Task 1: Codex専用画像変換境界を実装した。
- [x] 2026-08-20: Task 2: WD14語彙と画像由来tag追加validationを実装した。
- [x] 2026-08-20: Task 3: Codex scratch/process/gatewayへ画像添付を接続した。
- [x] 2026-08-20: Task 4: pipelineへ画像batch、cache、cleanup、hard-stopを統合した。
- [x] 2026-08-20: Task 5: GUI、metadata、security、利用文書を同期した。
- [x] 2026-08-20: Task 6: stale recovery test signatureをfix round 1で同期し、全non-live品質gateとFake E2Eを完了した。未認証live境界はskipとして記録した。

## Decision Log

- 2026-08-20: 原寸原fileではなく、metadata除去・長辺2048px以下の派生画像を送る。
- 2026-08-20: 1 callは最大8画像とし、contact sheetやbase64 JSONを使わない。
- 2026-08-20: Codexはpin済みWD14語彙内の不足tagを追加できる。
- 2026-08-20: Runtime Codex refinementでは画像送信を常時有効にし、project toggleや実行確認dialogを追加しない。
- 2026-08-20: 1枚でも画像を安全に確認できない場合は`allow_without_codex`に関係なくTraining前にrecoverable停止する。
- 2026-08-20: 派生画像はcall終了後に削除し、永続成果物へ画像本体やpathを残さない。
- 2026-08-20: Git操作禁止のため、writing-plans標準のcommit stepはすべて省略する。

### Task 1: Deterministic sanitized image attachments

**Files:**
- Create: `src/lora_factory/codex/image_attachment.py`
- Create: `tests/unit/test_codex_image_attachment.py`

**Interfaces:**
- Consumes: normalized working PNG `Path` and safe `asset_id`.
- Produces: `CodexImageProfile`, `PreparedCodexImage`, `prepare_codex_image(asset_id: str, source: Path, destination_root: Path, *, profile: CodexImageProfile = DEFAULT_CODEX_IMAGE_PROFILE) -> PreparedCodexImage`, `remove_codex_images(images: Sequence[PreparedCodexImage]) -> None`.

- [x] **Step 1: Write failing conversion and immutability tests**

```python
def test_prepare_codex_image_strips_metadata_resizes_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "working.png"
    create_oriented_profiled_png(source, size=(4096, 2048), exif_comment="private")
    before = sha256_file(source)

    first = prepare_codex_image("asset-a", source, tmp_path / "scratch-a")
    second = prepare_codex_image("asset-a", source, tmp_path / "scratch-b")

    assert (first.width, first.height) == (2048, 1024)
    assert first.byte_count <= 8 * 1024 * 1024
    assert first.output_sha256 == second.output_sha256
    assert first.profile == DEFAULT_CODEX_IMAGE_PROFILE
    assert sha256_file(source) == before
    with Image.open(first.path) as image:
        assert image.mode == "RGB"
        assert image.getexif() == {}
        assert "comment" not in image.info


def test_prepare_codex_image_rejects_unsafe_asset_id_and_output_escape(tmp_path: Path) -> None:
    source = make_png(tmp_path / "working.png", size=(64, 64))
    with pytest.raises(ValueError, match="asset_id"):
        prepare_codex_image("../escape", source, tmp_path / "scratch")
```

Add tests for no-upscale, quality fallback order `(95, 90, 85, 80)`, failure above 8 MiB after quality 80 using a monkeypatched encoder, atomic temporary cleanup, and `remove_codex_images()` idempotency.

- [x] **Step 2: Run the focused test and confirm the missing module failure**

Run: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_image_attachment.py -q`

Expected: collection fails because `lora_factory.codex.image_attachment` does not exist.

- [x] **Step 3: Implement the frozen Pydantic boundary and deterministic encoder**

```python
class CodexImageProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_edge: Annotated[int, Field(ge=256, le=4096)] = 2048
    max_bytes: Annotated[int, Field(ge=1024, le=16 * 1024 * 1024)] = 8 * 1024 * 1024
    jpeg_qualities: tuple[int, ...] = (95, 90, 85, 80)
    resampling: Literal["lanczos"] = "lanczos"


class PreparedCodexImage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    asset_id: str
    path: Path
    relative_name: str
    source_sha256: str
    output_sha256: str
    width: int
    height: int
    quality: int
    byte_count: int
    profile: CodexImageProfile
```

Validate the asset ID with the same safe filename alphabet as `RawTagStore`. Resolve the destination against `destination_root`, encode through `BytesIO` with fixed `format="JPEG"`, `subsampling=0`, `optimize=False`, `progressive=False`, and write bytes through a same-directory temporary file followed by `os.replace`. Open only the working image, convert to RGB, use `thumbnail((2048, 2048), Image.Resampling.LANCZOS)` without upscaling, and pass no EXIF/ICC/comment fields to `save()`.

- [x] **Step 4: Run focused tests and static checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_image_attachment.py -q
.venv\Scripts\ruff.exe check src\lora_factory\codex\image_attachment.py tests\unit\test_codex_image_attachment.py
.venv\Scripts\python.exe -m mypy src\lora_factory\codex\image_attachment.py
```

Expected: all commands exit 0; no task-created image remains outside pytest temporary directories.

### Task 2: WD14 vocabulary and image-derived tag validation

**Files:**
- Create: `src/lora_factory/caption/tag_vocabulary.py`
- Modify: `src/lora_factory/caption/tagger.py`
- Modify: `src/lora_factory/caption/refinement.py`
- Modify: `src/lora_factory/application/refinement_review.py`
- Modify: `tests/unit/test_caption_refinement.py`

**Interfaces:**
- Consumes: managed `selected_tags.csv`, Fake backend vocabulary, `CaptionDraftBatch`, Codex proposals.
- Produces: `WD14TagVocabulary.from_csv(path: Path)`, `WD14TagVocabulary.fake()`, `contains(tag: str) -> bool`, `batch_refinement_assets(assets, *, max_items=8, max_bytes=128 * 1024)`, `ValidatedAssetDecision.rejected_tags`, `RefinementReviewItem.added_tags`, `RefinementReviewItem.removed_tags`.

- [x] **Step 1: Write failing vocabulary, batch, and selective-rejection tests**

```python
def test_wd14_vocabulary_reads_utf8_bom_and_canonical_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text("name,category\nblue_hair,0\nrating:safe,9\n", encoding="utf-8-sig")
    vocabulary = WD14TagVocabulary.from_csv(csv_path)
    assert vocabulary.contains("blue hair")
    assert vocabulary.model_category("rating:safe") == "rating"


def test_image_proposal_keeps_known_addition_and_rejects_unknown_only() -> None:
    validated = validate_refinement_proposals(
        drafts,
        (proposal("asset-a", tags=("smile", "blue_hair", "invented_fact")),),
        vocabulary=WD14TagVocabulary.from_names(("smile", "blue_hair")),
    )
    assert validated.effective_tags["asset-a"] == ("smile", "blue_hair")
    assert validated.decisions["asset-a"].rejected_tags == {
        "invented_fact": "tag is absent from the pinned WD14 vocabulary"
    }


def test_refinement_batches_stop_at_eight_assets() -> None:
    batches = batch_refinement_assets(make_draft_assets(17), max_items=8)
    assert [len(batch) for batch in batches] == [8, 8, 1]
```

Add coverage for empty/duplicate CSV rows, rating/character categories, preset-forbidden known tags, separator rejection, max-tag overflow rejecting excess additions deterministically, stable asset ordering, and the simultaneous 8-item/128-KiB limits.

- [x] **Step 2: Run focused tests and confirm missing APIs**

Run: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_caption_refinement.py -q`

Expected: failures mention missing `WD14TagVocabulary`, `batch_refinement_assets`, and `rejected_tags`.

- [x] **Step 3: Implement shared vocabulary and selective tag validation**

```python
class WD14TagVocabulary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tags: frozenset[str]
    model_categories: dict[str, Literal["general", "character", "rating"]]

    @classmethod
    def from_csv(cls, path: Path) -> WD14TagVocabulary:
        categories: dict[str, Literal["general", "character", "rating"]] = {}
        with path.resolve(strict=True).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = normalize_tag_key((row.get("name") or row.get("tag") or "").strip())
                if not name or name in categories:
                    raise ValueError("WD14 CSV contains an empty or duplicate tag")
                category_id = (row.get("category") or "0").strip()
                categories[name] = (
                    "rating"
                    if category_id == "9"
                    else "character"
                    if category_id == "4"
                    else "general"
                )
        if not categories:
            raise ValueError("WD14 CSV is empty")
        return cls(tags=frozenset(categories), model_categories=categories)

    @classmethod
    def from_names(cls, names: Iterable[str]) -> WD14TagVocabulary:
        canonical = tuple(normalize_tag_key(name) for name in names)
        if (
            not canonical
            or any(not name for name in canonical)
            or len(set(canonical)) != len(canonical)
        ):
            raise ValueError("WD14 vocabulary names must be non-empty and unique")
        return cls(
            tags=frozenset(canonical),
            model_categories={name: "general" for name in canonical},
        )

    @classmethod
    def fake(cls) -> WD14TagVocabulary:
        return cls.from_names(FAKE_WD14_VOCABULARY)

    def model_category(self, tag: str) -> Literal["general", "character", "rating"] | None:
        return self.model_categories.get(normalize_tag_key(tag))

    def contains(self, tag: str) -> bool:
        return normalize_tag_key(tag) in self.tags
```

Move the Fake tag list to exported `FAKE_WD14_VOCABULARY` in `tagger.py` and have `FakeTagger` consume the same tuple. In `validate_refinement_proposals`, retain input tags, allow additions only when `vocabulary.contains(tag)`, remove only each invalid, duplicate, separator-bearing, forbidden, or over-limit added tag, and populate `rejected_tags` with a deterministic reason. Treat duplicate/missing/unknown assets and a `keep` decision that changes tags as response-contract failures. Compute `added_tags` and `removed_tags` from the validated effective list for review display.

- [x] **Step 4: Run caption/refinement tests and type checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_caption_refinement.py tests\unit\test_caption_policies.py -q
.venv\Scripts\ruff.exe check src\lora_factory\caption tests\unit\test_caption_refinement.py
.venv\Scripts\python.exe -m mypy src\lora_factory\caption
```

Expected: pass with deterministic ordering and no change to raw `TagScore` records.

### Task 3: Codex CLI image attachment boundary

**Files:**
- Modify: `src/lora_factory/codex/scratch_repo.py`
- Modify: `src/lora_factory/codex/process.py`
- Modify: `src/lora_factory/codex/gateway.py`
- Modify: `src/lora_factory/codex/prompts.py`
- Modify: `tests/unit/test_codex_gateway.py`
- Modify: `tests/unit/test_codex_static_schemas.py`

**Interfaces:**
- Consumes: `Sequence[PreparedCodexImage]` already located below the dedicated scratch root.
- Produces: `ScratchCall.image_paths`, repeated `--image <path>` from `build_codex_arguments(executable: str, call: ScratchCall, prompt: str)`, `CodexGateway.review(task_type, sanitized_input, *, images: Sequence[PreparedCodexImage] = (), allow_fallback: bool)`, audit fields `image_input_sha256s` and `image_count`.

- [x] **Step 1: Write failing argument, containment, prompt, and audit tests**

```python
def test_codex_arguments_attach_each_image_without_shell_string(tmp_path: Path) -> None:
    call = prepared_scratch_call(tmp_path, image_names=("asset-a.jpg", "asset-b.jpg"))
    arguments = build_codex_arguments(executable="codex", call=call, prompt="Review")
    attached = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--image"]
    assert attached == [str(path) for path in call.image_paths]
    assert len(attached) == 2


def test_scratch_rejects_image_outside_dedicated_root(tmp_path: Path) -> None:
    outside = make_jpeg(tmp_path / "outside.jpg")
    with pytest.raises(ValueError, match="scratch repository"):
        ScratchRepository(tmp_path / "runtime").prepare_call(
            call_id="unsafe",
            task_type=CodexTaskType.DATASET_REFINEMENT,
            sanitized_input=payload(),
            response_model=DatasetRefinementResponse,
            image_paths=(outside,),
        )
```

Add tests that dataset refinement rejects zero images, non-refinement tasks still accept zero images, image order is preserved, prompt identifies every relative image filename/asset ID pair, and audit contains hashes/count but no absolute paths.

- [x] **Step 2: Run focused tests and confirm signature/assertion failures**

Run: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_gateway.py tests\unit\test_codex_static_schemas.py -q`

Expected: failures for absent image fields and no `--image` arguments.

- [x] **Step 3: Extend the scratch/process/gateway contracts**

```python
@dataclass(frozen=True, slots=True)
class ScratchCall:
    root: Path
    input_path: Path
    schema_path: Path
    output_path: Path
    events_path: Path
    stderr_path: Path
    image_paths: tuple[Path, ...] = ()


def build_codex_arguments(*, executable: str, call: ScratchCall, prompt: str) -> list[str]:
    image_arguments = [item for path in call.image_paths for item in ("--image", str(path))]
    return [
        executable,
        "exec",
        *image_arguments,
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(call.schema_path),
        "--output-last-message",
        str(call.output_path),
        "--cd",
        str(call.root),
        prompt,
    ]
```

Resolve every image strictly, require `path.is_relative_to(scratch.root)`, require a regular file, reject symlinks/reparse escape through the existing resolved-path rule, and cap dataset-refinement images at 8. `CodexGateway.review()` must require at least one image for `DATASET_REFINEMENT`, pass the tuple to `prepare_call()`, and hash the bytes into `CodexAudit` without serializing paths. Update the prompt to state that visual facts may be added only from the pinned vocabulary represented by the Factory contract and that filenames map exactly to JSON asset IDs.

- [x] **Step 4: Run Codex boundary tests and static checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_gateway.py tests\unit\test_codex_static_schemas.py -q
.venv\Scripts\ruff.exe check src\lora_factory\codex tests\unit\test_codex_gateway.py
.venv\Scripts\python.exe -m mypy src\lora_factory\codex
```

Expected: pass; captured subprocess arguments contain repeated `--image`, remain a list, and contain only paths below the scratch root.

### Task 4: Pipeline image batches, cache, cleanup, and hard stop

**Files:**
- Modify: `src/lora_factory/application/service.py`
- Modify: `src/lora_factory/application/artifact_store.py`
- Modify: `tests/integration/test_application_fake_e2e.py`
- Modify: `tests/unit/test_foundation.py`

**Interfaces:**
- Consumes: `NORMALIZING.items`, accepted IDs from `DEDUPLICATING`, `CaptionDraftBatch`, `WD14TagVocabulary`, image attachment/gateway APIs.
- Produces: image-backed `CODEX_REFINEMENT` artifacts with `image_profile`, `image_sha256s`, `working_sha256s`, `added_tags`, `removed_tags`, cache records, progress events, and recoverable failures before `TRAINING`.

- [x] **Step 1: Write failing integration tests for all-image coverage and cleanup**

```python
def test_fake_refinement_prepares_every_accepted_image_in_batches_and_cleans_up(app) -> None:
    result = app.run_project(image_count=17, refinement_mode="auto")
    output = app.stage_output(result.run_id, PipelineStage.CODEX_REFINEMENT)
    assert [audit["image_count"] for audit in output["audits"]] == [8, 8, 1]
    assert sum(audit["image_count"] for audit in output["audits"]) == 17
    assert not list(app.codex_scratch.glob("input/images/**/*.jpg"))


def test_image_preparation_failure_blocks_training_even_when_fallback_allowed(
    app, monkeypatch
) -> None:
    monkeypatch.setattr(
        "lora_factory.application.service.prepare_codex_image",
        Mock(side_effect=ImageAttachmentError("cannot sanitize image")),
    )
    result = app.run_project(allow_without_codex=True)
    assert result.status == RunStatus.FAILED_RECOVERABLE.value
    assert app.training_attempts(result.run_id) == []
```

Add tests for working and Raw hash equality, two preparation attempts, gateway timeout/unavailable with `allow_without_codex=True`, asset/image mismatch, cache hit after restart, changed working hash invalidating only its batch, incomplete batch rerun, cancellation cleanup, and successful tag-only candidate failure entering Trigger Word review rather than Training.

- [x] **Step 2: Run focused integration tests and confirm current tag-only behavior fails**

Run: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\integration\test_application_fake_e2e.py -q`

Expected: new assertions fail because current `CODEX_REFINEMENT` creates no images and allows deterministic fallback.

- [x] **Step 3: Implement the batch orchestration with a single cleanup owner**

Build `normalized_by_asset` from `NORMALIZING.items`, assert exact coverage of every draft asset, and resolve the real vocabulary from `runtime.layout.wd14 / "selected_tags.csv"`; use `WD14TagVocabulary.fake()` for Fake backend. Store records as `codex-refinement/batch-{index:04d}.json`; do not reuse the old tag-only `chunk-*.json` records because they lack image fingerprints. Update existing integration assertions from `chunk-*` to `batch-*`. For each batch:

```python
image_root = self.settings.codex_runtime_root / "input" / "images" / batch_key
prepared: list[PreparedCodexImage] = []
try:
    prepared.extend(
        self._prepare_codex_images_with_two_attempts(batch, normalized_by_asset, image_root)
    )
    payload = self._refinement_payload(batch, prepared, vocabulary)
    input_hash = refinement_fingerprint(payload)
    cached = self._load_matching_refinement_batch(input_hash, prepared)
    response = cached or self._codex_review(
        context,
        CodexTaskType.DATASET_REFINEMENT,
        payload,
        images=prepared,
        allow_fallback=False,
    )
finally:
    remove_codex_images(prepared)
    remove_empty_image_root(image_root, boundary=self.settings.codex_runtime_root)
```

`_prepare_codex_images_with_two_attempts()` must collect each completed image immediately and call `remove_codex_images()` on its partial list before retrying or raising, so a failure on image N cannot orphan images 1..N-1. Do not place absolute paths in `payload`, batch records, events, DB audit, manifest, or package. Publish `codex_image_progress` before/after preparation and `codex_batch_progress` for call/retry/cache hit. Convert image preparation, missing Codex, auth, timeout, invalid schema, and mapping failures to `PipelineError("Runtime Codex image refinement failed: <sanitized reason>", recoverable=True)` after passing the reason through the existing redaction boundary. Preserve the current per-tag rejection behavior and Trigger candidate wait. Ensure `_codex_review()` keeps existing fallback behavior for every non-dataset-refinement task.

- [x] **Step 4: Run integration, resume, cancellation, and artifact-store tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\integration\test_application_fake_e2e.py tests\integration\test_archaudit_application_resume.py tests\unit\test_foundation.py -q
.venv\Scripts\python.exe -m mypy src\lora_factory\application
```

Expected: all pass; no `TrainingAttemptRow` exists on image/Codex failure, and no task-created JPEG remains.

### Task 5: GUI disclosure, review diffs, audit metadata, and documentation

**Files:**
- Modify: `src/lora_factory/gui/project_editor.py`
- Modify: `src/lora_factory/gui/dataset_review.py`
- Modify: `src/lora_factory/gui/main_window.py`
- Modify: `tests/gui/test_gui_workflow.py`
- Modify: `docs/architecture.md`
- Modify: `docs/user-guide-ja.md`
- Modify: `README.md`
- Modify: `SECURITY.md`
- Modify: `src/lora_factory/application/service.py`
- Modify: `tests/e2e/test_cli_fake_e2e.py`

**Interfaces:**
- Consumes: image progress events, `RefinementReviewItem.added_tags/removed_tags`, image audit hashes/profile.
- Produces: always-visible OpenAI image-upload disclosure, batch progress, distinct added/removed tag display, path-free reproducibility metadata and accurate security documentation.

- [x] **Step 1: Write failing GUI and metadata assertions**

```python
def test_project_editor_discloses_automatic_codex_image_upload(qtbot) -> None:
    editor = ProjectEditor()
    qtbot.addWidget(editor)
    notice = editor.findChild(QLabel, "codexImageUploadNotice")
    assert notice is not None
    assert "OpenAI" in notice.text()
    assert "2048" in notice.text()


def test_refinement_review_distinguishes_added_and_removed_tags(qtbot, review) -> None:
    review.set_refinement_review(review_payload(added=("blue_hair",), removed=("outdoors",)))
    text = review.refinement_table.item(0, review.CHANGE_COLUMN).text()
    assert "+ blue hair" in text
    assert "- outdoors" in text
```

Add assertions that progress renders `Preparing Codex images 8/17` and `Codex image batch 2/3`, manifest contains profile/hash/count but no `.jpg`, drive letter, home path, Raw/project path, or original filename, and CLI Fake E2E reports all accepted images as visually reviewed.

- [x] **Step 2: Run GUI/E2E focused tests and confirm missing disclosure/diff fields**

Run: `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\gui\test_gui_workflow.py tests\e2e\test_cli_fake_e2e.py -q`

Expected: new widget, diff, event, and manifest assertions fail.

- [x] **Step 3: Implement GUI and public-contract updates**

Add a word-wrapped `QLabel` named `codexImageUploadNotice` immediately below the Caption/Tag refinement controls:

```python
self.codex_image_upload_notice = QLabel(
    "Runtime Codex sends metadata-free, resized copies of every accepted image to OpenAI "
    "in batches of up to 8. Originals and project paths are not sent."
)
self.codex_image_upload_notice.setObjectName("codexImageUploadNotice")
self.codex_image_upload_notice.setWordWrap(True)
```

Render `added_tags` and `removed_tags` with explicit `+`/`-` prefixes without exposing paths. Map the new progress events through existing worker/controller event payloads, not direct service access from widgets. Add `codex_image_profile`, `codex_image_count`, ordered image hashes, batch input hashes, and cleanup status to reproducibility metadata; exclude image files and paths from packaging.

Update all four documents consistently: sanitized images are sent to OpenAI whenever Runtime Codex refinement runs; Raw/original images are not sent; metadata-free scratch JPEGs are deleted after each call; image/Codex failure blocks Training; other Codex review tasks retain their fallback behavior. Cite `https://developers.openai.com/codex/cli/reference/` for `--image`.

- [x] **Step 4: Run GUI, docs consistency, and fake CLI checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\gui\test_gui_workflow.py tests\e2e\test_cli_fake_e2e.py -q
rg -n "Codex.*画像.*受け取りません|Codex never sees images|raw images and secrets are not supplied|入力tagの.*部分集合" README.md SECURITY.md docs src\lora_factory
git diff --check
```

Expected: tests and `git diff --check` pass; the search returns no stale image-non-disclosure or subset-only contract.

### Task 6: Full verification and live image smoke test

**Files:**
- Modify: `tests/live/test_codex_live.py`
- Modify: this plan's `Progress`, `Validation`, and `Outcomes` sections.
- Modify: common Obsidian LoRA-Factory work log with a safe Japanese summary.

**Interfaces:**
- Consumes: completed image-backed refinement implementation.
- Produces: fresh focused/full verification evidence, optional authenticated live evidence, clean task artifacts, and an honest handoff without Git mutations.

- [x] **Step 1: Add the opt-in live image test**

```python
@pytest.mark.live_codex
def test_live_codex_dataset_refinement_reads_sanitized_image(tmp_path: Path) -> None:
    if os.environ.get("LORA_FACTORY_LIVE_CODEX") != "1":
        pytest.skip("set LORA_FACTORY_LIVE_CODEX=1 to call authenticated Codex")
    prepared = prepare_codex_image(
        "asset-a",
        make_non_personal_fixture(tmp_path / "fixture.png"),
        tmp_path / "runtime" / "input" / "images" / "live",
    )
    try:
        result = CodexGateway(tmp_path / "runtime").review(
            CodexTaskType.DATASET_REFINEMENT,
            live_refinement_payload(prepared),
            images=(prepared,),
            allow_fallback=False,
        )
        assert result.audit.image_count == 1
        assert result.response.assets[0].asset_id == "asset-a"
    finally:
        remove_codex_images((prepared,))
```

The fixture must be generated locally, contain no person or private data, and use only vocabulary included in its input contract.

- [x] **Step 2: Run focused suites, then formatting/lint/type checks**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_image_attachment.py tests\unit\test_caption_refinement.py tests\unit\test_codex_gateway.py tests\integration\test_application_fake_e2e.py tests\gui\test_gui_workflow.py -q
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m mypy src
```

Expected: every command exits 0.

- [x] **Step 3: Run the full suite and branch coverage gate**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider
.venv\Scripts\python.exe -m pytest -p no:cacheprovider --cov=lora_factory --cov-branch --cov-report=term --cov-fail-under=80
```

Expected: no failures and branch coverage at least 80%. Record exact pass/skip counts and coverage.

- [x] **Step 4: Run Character and Style Fake E2E**

Run:

```powershell
.venv\Scripts\python.exe -m lora_factory.cli fake-e2e
.venv\Scripts\python.exe -m lora_factory.cli fake-e2e --preset style --image-count 18 --json
```

Expected: both reach `READY`; every accepted image is accounted for in image-backed refinement audits; Raw and working hashes are unchanged; scratch image directories are empty afterward.

- [x] **Step 5: Run authenticated live Codex when available**

First run `codex login status`. If authenticated, run:

```powershell
$env:LORA_FACTORY_LIVE_CODEX='1'
.venv\Scripts\python.exe -m pytest -p no:cacheprovider -m live_codex tests\live\test_codex_live.py -q
Remove-Item Env:LORA_FACTORY_LIVE_CODEX
```

If not authenticated, do not weaken or bypass authentication; record the exact skip reason as an unverified live boundary.

- [x] **Step 6: Final diff, privacy, artifact, and Obsidian review**

Run:

```powershell
git diff --check
git status --short
rg -n "BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|api[_-]?key|access[_-]?token|refresh[_-]?token" src tests docs README.md SECURITY.md
rg -n "Codex.*画像.*受け取りません|Codex never sees images|raw images and secrets are not supplied|入力tagの.*部分集合" README.md SECURITY.md docs src\lora_factory
Get-ChildItem -Recurse -File -Filter *.jpg $env:LOCALAPPDATA\LoRAFactory\codex-scratch -ErrorAction SilentlyContinue
```

Review every secret-like match in context without printing actual secret values. Resolve the exact scratch root from application settings before deleting any task-created artifact; do not delete user data or broad directories. Verify the Obsidian note exists, retains its Japanese title/frontmatter/Wikilinks, contains actual test results, and has no terminal-specific duplicate. Do not stage, commit, push, merge, or create a PR.

## Validation

Expected results: deterministic metadata-free JPEGs; maximum 8 images per call; every accepted asset mapped exactly once; known WD14 additions accepted and invalid additions rejected individually; all-image failures block Training even with fallback enabled; cleanup succeeds on success/failure/timeout/cancel/cache hit; Raw and working hashes remain unchanged; Trigger Word and downstream artifacts remain consistent; all non-live quality gates pass.

### 2026-08-20 Task 6 actual results

- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\unit\test_codex_image_attachment.py tests\unit\test_caption_refinement.py tests\unit\test_codex_gateway.py tests\integration\test_application_fake_e2e.py tests\gui\test_gui_workflow.py -q`: exit 0, 89 passed in 47.62s.
- Initial `.venv\Scripts\ruff.exe format --check .` and `.venv\Scripts\ruff.exe check .` found only the new live test style issues and three Python code-fence formatting issues in this plan. After in-scope corrections, fresh runs exited 0: format reported 205 files already formatted; lint reported `All checks passed!`. Ruff emitted non-fatal `.ruff_cache` access-denied warnings. `.venv\Scripts\python.exe -m mypy src`: exit 0, no issues in 145 source files.
- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider`: exit 1, 674 passed, 6 skipped, 1 failed in 117.07s. The failing test is `tests/integration/test_archaudit_application_recovery.py::test_archaudit_application_oom_recovery_uses_fresh_attempt_and_advisory_codex`; its local `capture_review` monkeypatch has the pre-image signature and rejects the now-required `images=` keyword. This file is outside Task 6's edit allowlist, so it was not changed.
- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider --cov=lora_factory --cov-branch --cov-report=term --cov-fail-under=80`: coverage threshold reached at 83.24%, but command exit 1 because the same test failed; 674 passed, 6 skipped, 1 failed in 130.79s.
- `.venv\Scripts\python.exe -m lora_factory.cli fake-e2e`: exit 0, Character reached `READY`. `.venv\Scripts\python.exe -m lora_factory.cli fake-e2e --preset style --image-count 18 --json`: exit 0, Style reached `READY`. For each run, 18 accepted images were covered exactly once by three image-backed audits (`8,8,2`), audit image-hash counts were `8,8,2`, 18/18 working hashes matched the generated working PNGs, source hashes were unchanged, manifest image/hash counts were 18, cleanup status was `removed_after_each_call`, and scratch JPEG count was 0. The two Task 6 Fake E2E run roots were resolved below `.artifacts/fake-e2e` and removed after inspection.
- `codex login status`: exit 1 with `Not logged in`. `LORA_FACTORY_LIVE_CODEX` was not set, no live Codex call was made, and the live image boundary remains unverified. The opt-in test uses a locally generated solid-color non-person fixture, limits the accepted response vocabulary to the three tags represented in its sanitized input, and cleans the prepared JPEG in `finally`.
- `git diff --check`: exit 0. Secret-like search over `src tests docs README.md SECURITY.md`: exit 1 with zero matches. The original stale-contract search returned four self/history matches below `docs/superpowers/plans/**`; the binding adjusted search excluding that directory exited 1 with zero current public docs/source matches.
- `default_app_settings().codex_runtime_root` resolved to `C:\Users\mofu\AppData\Local\LoRAFactory\codex-scratch`; the required recursive JPEG scan returned zero files. No credentials were inspected or modified.

### 2026-08-20 Task 6 fix/verification round 1

- `tests/integration/test_archaudit_application_recovery.py`のtest-local `capture_review`へ、production `_codex_review`と互換のkeyword-only `images: Sequence[PreparedCodexImage] = ()`と`allow_fallback: bool | None = None`を追加した。recovery payload captureを維持し、両引数を変更せず`original_review`へ転送した。production codeとassertionは変更していない。
- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\integration\test_archaudit_application_recovery.py::test_archaudit_application_oom_recovery_uses_fresh_attempt_and_advisory_codex -q`: exit 0, 1 passed in 7.69s.
- `.venv\Scripts\ruff.exe format --check .`: exit 0, 205 files already formatted. `.venv\Scripts\ruff.exe check .`: exit 0, all checks passed. `.venv\Scripts\python.exe -m mypy src`: exit 0, no issues in 145 source files. Ruffの`.ruff_cache` access-denied warningは非致命で、終了コードは0だった。
- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider`: exit 0, 675 passed, 6 skipped in 111.55s. Skipはlive Codex 2件と未導入managed runtime 4件だけだった。
- `.venv\Scripts\python.exe -m pytest -p no:cacheprovider --cov=lora_factory --cov-branch --cov-report=term --cov-fail-under=80`: exit 0, 675 passed, 6 skipped in 120.61s; branch coverage 83.24%で80% gateを通過した。
- Fix round 1後の`git diff --check`はexit 0。secret-like privacy searchと`docs/superpowers/plans/**`を除くadjusted stale-contract searchは各exit 1、0 match。configured scratch rootは`C:\Users\mofu\AppData\Local\LoRAFactory\codex-scratch`で、recursive JPEG scanはexit 0、0件だった。
- Character/Style Fake E2Eはsignature test fixの影響範囲外なので再実行せず、直前roundのfresh evidence（両方`READY`、各18/18、audit `8,8,2`、Raw/working hash不変、JPEG 0件）を保持した。`codex login status`も再実行せず、直前roundの`Not logged in`に基づきauthenticated liveは引き続きskipであり、live passは主張しない。

## Outcomes

Initial Task 6 verification was blocked by one stale recovery-test monkeypatch signature. Fix/verification round 1 expanded the allowlist, synchronized that test-only signature without changing production behavior or assertions, and fresh targeted/full/coverage gates all passed. Task 1 through Task 6のnon-live acceptance criteriaは完了し、Character/Style image-backed Fake E2E evidenceも有効である。Authenticated live evidenceはCodex未loginのためskipのままで、live passは主張しない。全実装はbranch `v0.3`でunstaged/uncommittedのままであり、Git mutationは行っていない。

### 2026-08-20 Final whole-change review fix wave

- 6件のImportant findingを一体修正した。active Codex processは最大0.1秒間隔でcancelを検知し、即時terminate、既存5秒grace後killへ進む。controllerからgateway/processまでCancellationTokenを接続し、cancel時のretryとTrainingを停止する。
- Dataset refinementの全添付をprocess起動前に再検証する。prepared/payload/fileの件数・安定順・asset ID・相対名・source/output hash・profile・寸法・quality・byte数を一致させ、scratch containment、regular file、symlink/reparse拒否、read前後stat、実byte hash、RGB JPEG、実寸法を確認する。
- refinement approvalはatomic save前とresume時の両方で、pinned/Fake WD14語彙、ontology/category、separator、duplicate、最大tag数、Trigger/class/invariant/forbidden/semantic caption QAを通す。無効入力ではapprovalを保存せず`AWAITING_REVIEW`を維持し、Training attemptを作らない。
- terminal Codex failure/cancelはsanitized `CodexAudit`を持つ型付き例外となり、stage failureより先に`CodexCallRow`へ永続化する。stderr、path、filenameはauditへ保存しない。
- `validate_trigger_word()`を共通collision boundaryとして、manual config、候補normalize、review approval、final caption/pre-Trainingへ適用した。`1girl`等はGUIを含めてblocking errorになる。
- 現行public docs 4件を、pinned WD14からの視覚的addition許可・無効additionの個別拒否・候補3件未満時のmanual `AWAITING_REVIEW`へ同期し、historical planを除外するstale-contract regressionを追加した。

Fresh verification:

- focused 5-file pytest: exit 0, 463 passed in 56.71s。
- `tests/unit/test_codex_image_attachment.py`: exit 0, 12 passed in 0.61s。
- touched 16 Python filesのRuff format check: exit 0, 16 files already formatted。Ruff lint: exit 0, all checks passed。既知の`.ruff_cache` ACL warningのみ。
- touched 11 source filesのmypy: exit 0, no issues found。
- `git diff --check`: exit 0。secret-like searchとhistorical plan除外stale-contract search: 各exit 1、0 match。
- configured scratch root `C:\Users\mofu\AppData\Local\LoRAFactory\codex-scratch`: JPEG 0件。task-created Codex PID 3件は全て不存在。
- immutable pre-fix snapshotとの差分は意図した20 fileだけで、snapshotと`final-review.diff`は変更していない。Git mutationは行っていない。

Authenticated live CodexはTask 6時点の`Not logged in`により未検証のままであり、このfix waveのbriefに従ってfull suite/Fake E2Eは再実行していない。fresh focused/static/final gatesは全て成功した。

### 2026-08-21 運用開始前整合性監査の追加修正

- Windows Job Objectの`KILL_ON_JOB_CLOSE`とPOSIX process groupを共通`ProcessTree`へ追加し、ProcessManagerとCodex processのcancel/timeout/終了時に子孫を回収するよう修正した。親が孫を起動する回帰テストは修正前に孫残留で失敗し、修正後に通過した。
- Codex派生画像cleanupへscratch root containment、filename対応、symlink/reparse拒否を追加し、外部path forged recordの削除拒否を回帰テスト化した。Raw manifestのサイズはコピー元ではなくhash検証済みRaw destinationから記録するよう修正した。
- CodeQL workflowの浮動tagを、2026-08-21に公式GitHub APIで確認した`actions/checkout` v7.0.1および`github/codeql-action` v4のSHAへ固定した。publication preflightはcache overrideを尊重し、uv trampolineのWindows問題を避けてproject-local Pythonで品質ゲートを実行するよう修正した。
- 新修正後の全pytestは`696 passed, 6 skipped`、branch coverageは`83.29%`。publication preflightは`696 passed, 6 skipped`、coverage `83.25%`で成功。Ruff format/lint、mypy 146 source files、`uv --no-cache lock --check`、`uv --no-cache pip check`、Fake Character/Style E2Eの`READY`、Windows one-dir build、package static checks、起動スモークも確認した。
- この時点では実Codex未login、managed runtime未setup、sd-scripts manifest未確認だったため、実環境境界の判定を保留した。後続の2026-08-21再検証で状態を更新した。

### 2026-08-21 運用開始判定の最終検証

- 実ユーザー権限で`tests/live/test_managed_backends_live.py`を専用basetempから実行し、managed runtime doctorとWD14の`CUDAExecutionProvider`テストが`2 passed, 2 skipped`になった。skipは`LORA_FACTORY_LIVE_BASE_MODEL`未設定の学習・sampling 2件である。Codex sandboxユーザーでのDLL `WinError 5`は、runtime破損ではなく実行ユーザーのACL境界による再現差だった。
- 実ユーザー権限で`lora-factory doctor --deep --strict --json`を実行し、Codex Authentication、Managed Training Runtime、PyTorch CUDA/bf16、ONNX CUDA、pin済みsd-scriptsをすべて`ok`、runtime `ready: true`として検証記録へ更新した。TorchはCUDA 13.0、`sm_120`、bf16 smoke、ONNX CUDA transferを通過した。
- 認証済みCodexは利用可能だったが、画像を外部送信する`live_codex`は安全確認で明示許可が必要とされて停止した。迂回実行はせず、画像付き実Codex境界だけを未検証として残す。Fake/non-liveおよびmanaged GPU運用の判定には影響しない。
- 実ユーザー権限で`scripts/preflight_publication.ps1`を実行し、`699 passed, 4 skipped`、branch coverage `83.28%`、Ruff format/lint、mypy 146 source files、lock/pip check、Character Fake E2E `READY`を確認した。追加のStyle Fake E2E 18枚も`READY`、source hashes不変、`codex_refinement.codex_image_count=18`、batch `8,8,2`、JPEG残留0件だった。skipはCodex live 2件と、ベースモデル未設定のmanaged学習2件である。
- PyInstaller one-dirを再buildし、8秒GUI起動、必須license bundle 124件、重量ファイル0、秘密パターン0を確認した。起動時に空の`Project`ディレクトリが生成されることは、凍結版の既定project rootを実行ファイル隣に置く現行仕様・利用ガイド・回帰テストと一致するため、配布物混入とは判定しなかった。
- live/preflight/build用の専用一時ディレクトリ、外部cache、build出力、検証用PIDを確認後に削除・停止した。Codex scratchのJPEG残留は0件、`git diff --check`は成功し、stage/commit/pushは行っていない。

本監査の判定は「Fake/non-live運用可能」「managed runtime実GPU・WD14運用可能」「実Codex画像送信境界のみ、外部送信の明示許可待ち」である。Codex liveを除く運用開始条件は満たした。
