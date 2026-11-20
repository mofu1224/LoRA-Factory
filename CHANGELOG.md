# Changelog

All notable changes to LoRA Factory are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases use semantic versioning.

## [Unreleased]

### Changed

- 設定と学習環境をアプリフォルダー内に保存し、移動後はSetupの`Repair`から環境を再構築できます。
- 画像読み込みにPillow 13を使用します。既存の学習環境を上書きせず、新しい互換プロファイルをSetupから導入します。

- AUTO mode now retries Runtime Codex within one bounded budget, falls back to validated source tags after timeout/unavailable/invalid responses, auto-approves only safe proposals, and generates a stable local Trigger Word when no candidate is available. Explicit REVIEW and unresolved semantic coverage remain manual safety gates; existing AUTO waiting runs resume on GUI startup.

## [v0.3-bata] - 2026-08-27

### Changed

- Trigger Word now preserves any user-entered value and automatically uses the first validated Runtime Codex candidate only when the field is blank; candidate shortages still pause safely before training.
- Runtime Codex now detects Direct/Router installations, drains CLI pipes continuously, shares one total timeout across retries, and switches to an isolated profile only after Router/unknown startup failure.
- Windows GUI startup now pins the bundled PySide6 Qt DLLs before import and excludes build-host ICU DLL collisions that caused QtWidgets WinError 127.

### Security

- Added bounded image enumeration, byte-size, dimension, and pixel-count checks before import and full decoding.
- Removed local paths, original filenames, internal command arguments, and physical GPU UUIDs from final package metadata.

## [v0.1] - 2026-08-11

### Added

- Native Windows GUI for immutable image import, dataset review, WD14 tagging, Character/Style captioning, SDXL LoRA planning and training, sampling, evaluation, selection, packaging, destination copy, cancellation, and resume.
- Managed pinned backend runtime, selected-GPU isolation, deterministic Fake backend, and publication-grade Windows packaging.
