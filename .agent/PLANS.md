# ExecPlan discipline

An ExecPlan is the living, self-contained implementation record for work that takes more than about fifteen minutes. A new contributor must be able to continue from the plan without hidden context.

Each ExecPlan must contain and continuously maintain:

- `Purpose and acceptance`: the observable user outcome and hard constraints.
- `Progress`: timestamped checkboxes for completed, active, and remaining work.
- `Surprises & Discoveries`: unexpected runtime, dependency, platform, or test facts with evidence.
- `Decision Log`: decisions, rationale, alternatives rejected, and the date.
- `Architecture and milestones`: module boundaries, data flow, and independently verifiable increments.
- `Validation`: exact commands, expected observations, and actual outcomes.
- `Outcomes & Retrospective`: delivered behavior, remaining environment-dependent checks, and lessons.

Plans describe behavior, not merely file lists. Update them after each milestone and whenever evidence changes a decision. Do not erase useful history; mark superseded assumptions and explain why. A plan may record unfinished work, but production code may not disguise unfinished behavior with TODOs, `pass`, or `NotImplementedError`.
