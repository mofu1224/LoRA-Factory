# Runtime Codexによるタグ・caption調整とTrigger Word設計

## 目的

LoRA FactoryのGUIパイプラインで、Runtime Codexが全採用画像の学習用タグとcaptionを安全に調整し、ユーザーがTrigger Wordを設定した後、学習・評価・packagingまで完結できるようにする。

Runtime Codexは専用scratch Git repositoryのread-only sandboxで動作する。全採用画像について、Rawではなくmetadata除去・縮小・再encodeした一時派生画像をCodex CLIの画像入力へ添付する。Raw画像、Rawパス、project path、認証情報、GPU制御、学習実行権限は渡さない。Codexの出力はPydantic schemaとFactory側のallowlist・caption規則で再検証し、派生データにだけ適用する。

## 確定した利用者要件

- 操作入口はGUIパイプラインとし、独立した本番学習CLIコマンドは今回の主目的にしない。
- Codexによるタグ・caption変更は全採用画像を対象にする。
- Runtime Codex refinementを使用するrunでは、全採用画像のsanitized派生画像を常にOpenAIへ送信し、画像なしのtag-only処理へ自動fallbackしない。
- Codex用画像はmetadataを除去し、長辺2048px以下へ縮小する。原寸原fileは送信しない。
- Codex callはasset ID安定順の最大8画像単位とする。
- Codexは画像から確認でき、pin済みWD14語彙に存在する不足タグを追加できる。
- 適用方法はプロジェクト設定で、自動適用と確認後適用を切り替えられる。
- Trigger WordはCodexが3〜5候補を生成し、ユーザーが選択または編集できる。
- Trigger Wordを手入力する既存フローも維持する。
- 手入力Trigger Wordと自動適用の組合せでは、確認停止なしに完成まで進める。
- Codex候補の選択またはcaption差分確認が必要な場合は、学習前に安全停止し、承認後に同じrunを再開する。

## 採用方式

caption生成とDataset Reviewの間に専用のCodex調整機能を追加する。既存Dataset OverrideへCodex提案を書き込んでパイプライン全体を再実行する方式は、二重実行と状態の分かりにくさから採用しない。Codexにcaptionファイルを直接編集させる方式は、read-only境界と構造化検証を弱めるため採用しない。

## パイプライン

論理的な順序は次のとおりとする。

1. `TAGGING`
2. triggerなしcaption下書き生成
3. `CODEX_REFINEMENT`
4. 必要なら`AWAITING_REVIEW`
5. Trigger Wordと有効タグからcaption確定
6. `DATASET_REVIEW`
7. `PLANNING`
8. 既存のpreflight、training、sampling、evaluation、selection、packaging

triggerなしcaption下書きは、既存のCharacter/StyleポリシーからTrigger Word挿入だけを分離して生成する。Characterのclass tokenとidentity invariant、Styleの禁止カテゴリはこの時点から保持する。確定captionではTrigger Wordを必ず先頭に挿入し、既存caption QAを通す。

`CODEX_REFINEMENT`はasset IDの安定順で全採用画像を処理する。1 callは最大8 assetとし、同時にsanitized JSONを128 KiB以下へ抑える。どちらかの上限へ先に達した位置で決定論的にbatchを分割する。各batchの画像はasset IDだけのfile名でscratchへ置き、JSON内のasset IDと相互照合して個別の`--image`引数で添付する。完了batchは変換設定、working画像hash、派生画像hash、JSON hashが一致するときだけ再利用し、未完了batchだけ再実行する。

Codex用派生画像は、既存のorientation補正・sRGB変換・alpha合成済みworking画像から生成する。sRGB RGB、長辺最大2048px、upscaleなし、metadataなしのJPEGとし、品質を95、90、85、80の順で下げて8 MiB以下になる最初の結果を採用する。品質80でも8 MiBを超える場合は変換失敗とする。encoder引数を固定し、同じ入力bytesと設定から同じ出力bytesを得る。

## 設定モデル

プロジェクト設定へ次の列挙値を追加する。

- Codex調整モード: `auto`、`review`
- Trigger Wordモード: `manual`、`codex_suggest`

`manual`では有効なTrigger Wordを開始前に要求する。`codex_suggest`では開始時のTrigger Wordを任意とし、候補確定前にはcaption確定以降へ進めない。既存schema versionのプロジェクトは`manual`として読み込み、現在のTrigger Tokenをそのまま保持する。

設定と確定したTrigger Wordはrun snapshotへ保存する。resumeはsnapshotを正本とし、編集中のProject Editor値を暗黙に取り込まない。

## Codex入出力契約

`dataset_refinement` Codex taskのstrict schemaを画像入力へ拡張する。入力JSONにはPreset、ontology version、batch情報、asset ID、対応する相対画像file名、working画像hash、派生画像hash、変換profile、正規化済み元タグとconfidence、triggerなしcaption下書き、既知の警告だけを含める。各派生画像は`codex exec --image`で個別添付する。Raw画像、元file名、Raw/project path、GPU UUID、認証情報は含めない。

各画像の出力は次の値を必須とする。

- `asset_id`
- `decision`: `keep`または`replace`
- `effective_tags`
- `reason`
- `confidence`

call単位の出力にはTrigger Word候補を含められる。全batch完了後、重複排除した3〜5候補へ統合する。候補には候補文字列と短い理由を含める。

Codexは画像と元タグを比較し、誤検出タグの削除、canonicalization、重複除去、並べ替え、画像から直接確認できる不足タグの追加を行える。追加タグはpin済みmanaged WD14の`selected_tags.csv`に存在する語彙だけを許す。WD14の元タグ・confidenceは変更せず、Codexの結果は別の`effective_tags`として保持する。追加・削除は元タグとの差分からFactory側で算出し、理由とconfidenceとともに監査する。

## Factory側の再検証

Codex応答はschema検証後、次の規則で再検証する。

- 対象assetが重複なく1回ずつ返されている。
- 未知assetがない。
- `effective_tags`の各値は、対応assetの入力タグまたはpin済みWD14 `selected_tags.csv`の既知語彙に含まれる。
- タグはontologyでcanonicalizeし、区切り文字、空値、重複、最大数を拒否する。
- Characterのsource、artist、character、copyright、style、quality、rating、resolution、watermarkカテゴリを拒否する。
- Characterの安定class tokenとidentity invariantを既存ポリシーどおり扱う。
- Styleのartist、style、copyright、character、quality、rating、resolution、watermarkカテゴリを拒否する。
- Trigger Wordは空値、64文字超、caption区切り、反復空白、一般的なDanbooruタグとの衝突を拒否する。
- 最終captionはTrigger Wordが先頭で、class token位置、禁止カテゴリ、重複、semantic coverageを含む既存QAを通る。

個別の新規タグが無効な場合は、そのタグだけを破棄し、同じassetの検証済み変更を保持する。元タグの不正な変更、asset欠落、重複、未知asset、画像file名・hash不一致、schema不正はbatch全体の失敗とする。

## 永続化と監査

次の情報を区別して保存する。

- WD14元タグとconfidence
- triggerなしcaption下書き
- Codex callごとの入力hash、schema hash、応答hash、version、attempt、timeout、fallback状態
- working画像hash、派生画像hash、変換profile、batch内asset対応
- Codex提案
- Factory検証結果と拒否理由
- 自動採用またはユーザー承認した有効タグ
- ユーザーによるcaption手修正
- Trigger Word候補と確定値
- 承認対象の上流fingerprint

最終captionだけを学習用`.txt`へmaterializeする。Raw Store、working画像、WD14元結果、過去のユーザーoverrideは上書きしない。Codex用派生画像は各callの成功、失敗、timeout、cancel後に削除し、resume時はworking画像から決定論的に再生成する。reproducibility manifestとtraining metadataには調整モード、確定Trigger Word、画像変換profile、画像・batch audit hash、承認fingerprintを画像本体、秘密情報、file名、絶対パスなしで記録する。

## GUI

Project Editorへ次を追加する。

- Caption / Tag調整: 自動適用、確認して適用
- Trigger Word: 手入力、Codex候補から選択

Runtime Codex refinementは常に画像を添付し、project単位の画像送信toggleや実行ごとの確認dialogは設けない。Project Editor、実行画面、利用ガイドには、metadata除去・縮小済みの派生画像がOpenAIへ送信されることを明記する。進捗にはCodex用画像の準備数、現在batchと総batch、再試行、cache再利用、画像入力によるrecoverable停止を表示する。

既存Trigger Token入力欄はTrigger Word入力欄として維持し、`manual`では必須、`codex_suggest`では候補選択後の編集欄として使う。設定はproject再読込時に復元する。

Dataset ReviewはCodex調整結果を表示できるよう拡張する。各行で元タグ、追加タグ、削除タグ、最終提案タグ、元caption下書き、提案caption、理由、confidenceを区別して表示し、個別の採用、却下、手修正と一括採用を提供する。Trigger Word候補は選択後も編集でき、同じvalidatorを通す。

動作matrixは次のとおりとする。

| Trigger Word | 調整モード | 動作 |
|---|---|---|
| 手入力 | 自動 | 停止せず完成まで実行 |
| Codex候補 | 自動 | 候補選択だけ待ち、確定後は完成まで実行 |
| 手入力 | 確認 | 全画像差分の承認後に実行 |
| Codex候補 | 確認 | 候補選択と全画像差分の承認後に実行 |

## 確認待ちとresume

run statusへ`AWAITING_REVIEW`を追加する。これは失敗でもキャンセルでもなく、ユーザー入力を待つ正常なdurable状態である。確認待ちへ入る前にCodex子プロセスを終了し、workerとGPU leaseを残さない。

Recent Projectsは確認待ちを明示し、該当projectを開くと保存済み提案と候補を復元する。「承認して学習を続行」は決定内容をatomic保存し、保存された上流fingerprintが現在値と一致する場合だけ同じrunを再開する。不一致なら古い承認を拒否し、Codex調整から再計算する。

未確定Trigger Word、未承認のreview mode、caption QA不合格のいずれかがある間はTraining stageを開始しない。Trigger Wordだけを変更した場合は、taggingやCodex調整を再実行せず、caption確定と下流fingerprintだけを無効化する。

## エラーとfallback

画像decode・変換・byte上限・scratch containment・添付、Codex CLI未導入、未認証、timeout、異常終了、asset対応不一致、無効な構造化出力は最大2 attemptを使う。再試行後も失敗した場合、この`dataset_refinement` taskでは`allow_without_codex`に関係なくrecoverable failureとして停止する。画像なしのtag-only fallback、失敗assetの除外、元タグだけでのTraining続行は行わない。他のCodex taskの既存fallback policyは変更しない。

有効なasset応答を取得できたがTrigger Word候補だけが3件未満の場合、`codex_suggest`は`AWAITING_REVIEW`で手入力を求める。個別の追加タグだけが語彙・category検証に失敗した場合は、そのタグを破棄して監査理由を残し、batch全体を失敗させない。

キャンセル時はCodex子プロセスをterminateし、猶予後も残る場合はkillする。派生画像は成功、失敗、timeout、cancelの全経路で削除する。完了batchは入力hash一致時だけ再利用する。確認待ち以前には学習process、optimizer state、checkpointを作らない。

## テスト

### Unit

- 新設定モデル、相互制約、旧projectの互換読込
- triggerなしcaption下書きと最終caption生成
- Trigger Word候補の正規化、重複排除、衝突拒否
- metadata除去、orientation、sRGB、alpha合成、長辺2048px、8 MiB、決定論的JPEG変換
- Raw・working画像を変更しないこととscratch containment
- 最大8画像かつJSON 128 KiB以下の決定論的batch分割と統合
- `--image`を個別に渡すshellなしargument array
- asset欠落、重複、未知asset、画像file/hash不一致の拒否
- `selected_tags.csv`既知語彙の新規追加、未知語彙・禁止タグの個別拒否
- 個別tag拒否とbatch単位failure
- 新Pydantic response modelと静的JSON schemaのparity

### Integration

- 手入力＋自動適用が停止せず`READY`
- Codex候補＋自動適用が`AWAITING_REVIEW`後に`READY`
- review modeがTraining前に必ず停止する
- 承認後の同一run resume
- 再起動後の確認待ち復元
- stale fingerprint承認の拒否
- 完了batch再利用と未完了batch再実行
- 成功、失敗、timeout、cancel、cache hit後の派生画像cleanup
- 画像準備・添付・timeout・異常応答が`allow_without_codex`でもTrainingを開始しないこと
- 未確定Trigger WordまたはQA不合格でTrainingへ進まない

### GUI

- 2つの設定切替とproject再読込
- Trigger Word候補の選択・編集・validation
- 全画像before/after、理由、confidence表示
- 追加・削除tagの区別、画像送信の明示、画像準備・batch進捗
- 個別採用、却下、手修正、一括採用
- 無効状態で続行buttonが無効になる
- Recent Projectsの確認待ち表示と再開

### Fake E2Eと品質gate

CharacterとStyleで4設定組合せを検証し、全採用画像の派生画像生成、最大8枚batch、Raw・working hash不変、未承認または画像入力失敗時のTraining遮断、最終caption・Dataset TOML・成果物metadataのTrigger Word一致、cleanup、cancel/resume、決定論的再実行を確認する。

完了前に次を実行する。

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run pytest --cov=lora_factory --cov-branch --cov-report=term
uv run --frozen lora-factory fake-e2e
uv run --frozen lora-factory fake-e2e --preset style --image-count 18 --json
```

認証済みCodex CLIが利用できる場合は、個人情報を含まない小型画像fixtureを`--image`添付する`uv run pytest -m live_codex`も実行する。利用できない場合は未実施理由を報告する。

## 完了条件

- GUIで選択したCodex調整モードとTrigger Wordモードが保存・復元される。
- Runtime Codexが全採用画像のsanitized派生画像を直接確認し、pin済みWD14語彙の不足タグを追加でき、検証済みの有効タグとcaptionだけが学習へ渡る。
- Trigger Word候補を選択・編集でき、確定値がcaption、sampling、metadata、成果物で一貫する。
- 自動モードは必要なTrigger Word確定後、学習・評価・packagingまで完結する。
- 確認モードは承認前にTrainingへ進まず、再起動後も同じ確認状態を復元できる。
- 画像準備・添付・Codex失敗時はTrainingへ進まず、cancel、crash、resume、cleanupがRaw・working画像不変性を損なわない。
- 静的schema、ドキュメント、テスト、Fake E2Eが新しい契約と一致する。
