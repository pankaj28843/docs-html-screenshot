# AGENTS.md (docs-html-screenshot)

## Scope
- Follow `.github/copilot-instructions.md` and README.
- Keep CLI behavior simple; avoid new abstraction layers.
- Avoid backward compatibility unless explicitly requested.

## Workflow
- Use `uv run` for all Python commands.
- No silent error handling; let real failures surface.
- Do not add summary reports unless explicitly requested.

## Validation
- Run: `uv sync --extra dev`
- Run: `uv run ruff format . && uv run ruff check --fix .`
- Run: `uv run pytest tests/ -v`
- Run: `uv run docs-html-screenshot --help`

## Planning
- Store PRP plans in `~/codex-prp-plans` (not in-repo).
- Update the plan after each phase; keep UTC timestamps with `Z` suffix.

## Privacy / Safety
- Do not include local machine details, IPs, or tenant-specific data in code or docs.
- Avoid embedding local paths or runtime secrets in docs/examples.
