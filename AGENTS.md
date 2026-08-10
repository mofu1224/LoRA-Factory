# LoRA Factory repository rules

- Complex work follows `.agent/PLANS.md` and keeps the active ExecPlan current.
- Use Python 3.12 in a project-local `uv` environment. CUDA 13.1 is the host baseline; managed GPU wheels may use the closest officially published compatible CUDA runtime and must be capability-tested.
- Treat source images and every project `dataset/raw/` directory as immutable. Import by verified copy; never rename, overwrite, delete, caption, crop, resize, or hardlink raw assets.
- Do not modify pinned `sd-scripts` source in `vendor` or a managed backend runtime. Integrate it through adapters.
- Launch external programs with argument arrays. `shell=True`, `os.system`, and unvalidated user strings in commands are forbidden.
- The PySide6 main thread must not perform heavy work. GUI code calls Application Services only; Core remains headless-testable.
- Use Pydantic models at configuration and process boundaries.
- Every pipeline stage must persist state and support idempotency, cancellation, and safe resume.
- GPU UUIDs are the source of truth. A process may see only GPUs selected by the user; never use an unselected GPU.
- Runtime Codex works only in its dedicated scratch Git repository with a read-only sandbox. Revalidate structured output and apply only allowlisted changes.
- Preserve exceptions and actionable logs; never silently ignore failures.
- Add a regression test with each bug fix. Before completion run format, lint, type checking, unit, integration, GUI, and fake E2E checks.
- Do not declare TODOs, `pass`, or `NotImplementedError` to be a finished implementation.
- Fake backends exist for deterministic tests; production-capable real Tagger, Trainer, Sampler, and Codex adapters are required.
- Preserve user data and unrelated changes. Do not push to GitHub unless explicitly requested.
