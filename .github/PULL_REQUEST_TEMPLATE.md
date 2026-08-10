## Summary

Describe the user-visible change and its scope.

## Validation

- [ ] `uv lock --check`
- [ ] `uv run ruff format --check .`
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src`
- [ ] Tests and relevant E2E checks pass
- [ ] No source images, models, credentials, local paths, GPU UUIDs, or generated project data are included

## Safety

Explain effects on immutable Raw data, subprocess arguments, GPU isolation, resume behavior, and public metadata when applicable.
