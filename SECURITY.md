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
- Runtime Codex receives allowlisted structured metadata in a dedicated read-only scratch repository; raw images and secrets are not supplied.
- Final package metadata removes absolute paths, original filenames, command arguments, and physical GPU UUIDs.
- Model files and third-party runtimes remain subject to their own provenance and license requirements.
