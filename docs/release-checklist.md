# Release checklist

## Before the first public push

- Confirm `git status --short --ignored` contains no unexpected public candidates.
- Confirm the repository has no remote pointing at an unintended destination.
- Run `scripts/preflight_publication.ps1`.
- Review the complete staged diff and `git diff --cached --check`.
- Confirm the commit author email is a GitHub noreply address if personal-email privacy is required.
- Push only the `main` branch. Never use `git push --mirror`; local `refs/codex/*` are not publication refs.
- Enable private vulnerability reporting, branch protection, required CI, secret scanning, and Dependabot alerts in repository settings.

## Release

1. Confirm CI passes on `main`.
2. Run the Windows release workflow manually once and test the downloaded artifact on a clean Windows account.
3. Verify the ZIP SHA-256, launch `LoRA Factory.exe`, run Setup, and complete a Fake Backend project.
4. Create an annotated semantic-version tag such as `v0.1` and push that tag.
5. The release workflow builds and attaches the Windows ZIP and SHA-256 file.
6. Confirm the public build was created with `-SystemVcRuntime` and contains no MSVC DLLs.
7. Confirm the exact Qt/PySide source archive and LGPL source-offer procedure recorded in `licenses/Qt/README.md` remain available.
8. Confirm `install_vcredist.cmd` is in the ZIP and its no-WinGet fallback downloads and verifies the official Microsoft x64 Redistributable installer.
9. Review release notes and verify all bundled licenses before marking the release public.

Real Backend validation additionally requires an authenticated Codex CLI, a compatible NVIDIA GPU, the managed runtime, a user-supplied compatible base model, and a non-private test dataset.
