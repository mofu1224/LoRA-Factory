# Changelog

All notable changes to LoRA Factory are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases use semantic versioning.

## [Unreleased]

### Security

- Added bounded image enumeration, byte-size, dimension, and pixel-count checks before import and full decoding.
- Removed local paths, original filenames, internal command arguments, and physical GPU UUIDs from final package metadata.

## [v0.1] - 2026-08-11

### Added

- Native Windows GUI for immutable image import, dataset review, WD14 tagging, Character/Style captioning, SDXL LoRA planning and training, sampling, evaluation, selection, packaging, destination copy, cancellation, and resume.
- Managed pinned backend runtime, selected-GPU isolation, deterministic Fake backend, and publication-grade Windows packaging.
