# Security policy

## Supported versions

Security fixes are provided for the latest published release and the current `main` branch.

## Reporting a vulnerability

Use the repository's private **Security** → **Report a vulnerability** form. Do not open a public Issue for an unpatched vulnerability.

Include the affected version, impact, minimal reproduction, and suggested mitigation if known. Remove source images, model weights, credentials, access tokens, account details, personal filenames, absolute paths, and physical GPU UUIDs from every attachment.

Maintainers will acknowledge a complete report as soon as practical, validate it privately, and coordinate remediation and disclosure. No guaranteed response or fix timeline is promised.

## Security boundaries

- Source images and each project's `dataset/raw/` are immutable inputs.
- External commands use argument arrays without a shell.
- Runtime Codex uses read-only ephemeral execution, disables configured and known optional MCP/tool surfaces, drains stdout/stderr continuously, and applies bounded startup, idle, total, cancellation, and process-tree cleanup controls.
- Runtime Codex dataset refinement receives allowlisted structured metadata and metadata-free, resized JPEG copies of every accepted image, in batches of at most eight. Raw/original images, original filenames, project paths, and secrets are not supplied. Scratch JPEGs are deleted after every call.
- Codex may add visually supported tags only from the pinned WD14 vocabulary. Unknown, forbidden, duplicate, separator-bearing, and over-limit additions are rejected individually before captions or Training can consume them.
- Dataset image preparation or Codex refinement failure stops Training recoverably; other Codex review tasks retain their configured fallback behavior.
- All Runtime Codex payloads are sanitized immediately before gateway/scratch persistence; recovery receives aggregate GPU capacity and a portable base-model hash, never physical GPU UUIDs or local model paths.
- Managed child processes receive an explicit environment allowlist; parent credentials, proxy settings, and other unlisted variables are not inherited.
- Dataset scanning and immutable import reject symbolic links, junctions, and other filesystem reparse points before resolving or copying image files.
- Dataset input is capped at 2,000 images, matching the bounded pairwise duplicate/embedding comparison budget; embedding similarity is computed in blocks to avoid unbounded memory use.
- Trigger Words that collide with common Danbooru tags and invalid review edits are rejected before atomic approval save and again before Training.
- Final package metadata removes absolute paths, original filenames, command arguments, physical GPU UUIDs, and scratch image files while retaining only the image profile, counts, hashes, batch input hashes, and cleanup status needed for reproducibility; the base-model digest is format- and file-verified before publication.
- The image attachment uses the Codex CLI [`--image` reference](https://developers.openai.com/codex/cli/reference/).
- Model files and third-party runtimes remain subject to their own provenance and license requirements.
