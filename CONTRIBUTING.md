# Contributing

LoRA Factory welcomes focused bug fixes, tests, documentation corrections, and compatibility improvements.

## Development setup

Use Windows 11, Python 3.12, Git, and a project-local uv environment.

```powershell
.\scripts\bootstrap.ps1
```

Before opening a pull request, run the checks in [the developer guide](docs/developer-guide.md#quality-gates). Bug fixes require a regression test.

## Safety rules

- Never commit datasets, source images, captions, model weights, generated LoRA packages, runtime downloads, credentials, local paths, or physical GPU UUIDs.
- Never modify a user's source image or `dataset/raw/` object.
- Do not introduce `shell=True`, `os.system`, command strings, or execution of unvalidated user input.
- Keep the GUI main thread free of heavy work and keep Core headless-testable.
- Do not modify the pinned `sd-scripts` source; integrate through adapters.
- Preserve cancellation, persisted state, idempotency, and safe resume behavior.

## Pull requests

Keep changes scoped, describe user-visible behavior, list validation evidence, and update documentation when the workflow changes. By contributing, you agree that your contribution is licensed under this repository's MIT License.
