# AI Coding Agent Instructions for docs-html-screenshot

> **WARNING: This file contains instructions for AI coding assistants ONLY.**
> **DO NOT use these guidelines for user-facing documentation.**
> **For project documentation, see README.md.**

**Project Context**: CLI tool using Playwright to generate full-page screenshots of static HTML files (e.g., MkDocs output) with error detection and concurrent processing.

## Core Philosophy

**NO BACKWARD COMPATIBILITY** - This is an active development project. Break things freely unless explicitly asked to maintain compatibility. Focus on making the code better, not preserving old patterns.
**LESS CODE, NO BACKWARD COMPATIBILITY** - Default to deleting legacy branches and feature flags; every refactor should leave fewer lines than before unless there is a compelling reason otherwise.
**LEAST AMOUNT OF CODE NEEDED FOR FUNCTIONALITY** - Ship the minimal implementation that satisfies the requirement, avoid abstraction layers until repeated use proves they are needed.
**RESILIENT SOLUTIONS, BUT NEVER SILENCE ERRORS** - Harden workflows against expected failure modes, yet let exceptions bubble so we immediately see real regressions instead of masking them with broad `try/except` blocks.
**NEVER LEAVE FAILING TESTS BEHIND** - All unit tests MUST pass after every change. If you encounter a failing test:
1. Fix it if it's testing valid behavior that your changes broke
2. Update it if it's testing outdated behavior that no longer applies
3. Delete it if it's testing removed functionality

**NO SUMMARY REPORTS - UNLESS EXPLICITLY REQUESTED** - Do NOT create summary documents, reports, or "what we've accomplished" write-ups unless the user explicitly requests documentation.

## Prime Directives (ALWAYS)

- Use `uv run` **before any Python command**: `uv run pytest`, `uv run docs-html-screenshot`, etc.
- **Green-before-done**: Do not say "done" until edited Python files import cleanly and tests are green.
- **Tests are mandatory** for any change affecting code paths
- **Never hallucinate**: do not invent files, paths, models, or settings. Search the repo first.
- **TechDocs-first research**: Always check TechDocs before implementing (see workflow below)
- **Verify docs with real commands**: Before documenting any command, run it and paste actual output. Never invent "Expected output" blocks.

## TechDocs Research Workflow

**ALWAYS start with this sequence**:
1. `mcp_techdocs_list_tenants()` → discover available documentation sources
2. `mcp_techdocs_describe_tenant(codename="...")` → understand tenant capabilities and test queries
3. `mcp_techdocs_root_search(tenant_codename="...", query="...")` → find relevant patterns
4. `mcp_techdocs_root_fetch(tenant_codename="...", uri="...")` → read full documentation

**Key tenants for this project**: `playwright` (browser automation), `python`/`pytest` (Python best practices), `docker` (containerization), `github-platform` (GitHub Actions/CI)
**Full TechDocs guide**: See `.github/instructions/techdocs.instructions.md`

## Validation & Testing

**ALL code changes require validation loop**:
```bash
uv sync --dev
uv run ruff format . && uv run ruff check --fix .
uv run pytest tests/ -v
```

**Full validation checklist**: See `.github/instructions/validation.instructions.md`

## Testing Rules

- Unit tests for EVERY new function/class
- Test behavior, not implementation
- Use `pytest-asyncio` for async tests
- Mark async tests with `@pytest.mark.asyncio`

**Pytest patterns**: `.github/instructions/tests.instructions.md`

## Planning & Memory

For non-trivial tasks, create a PRP plan at `.github/ai-agent-plans/{ISO-timestamp}-{slug}-plan.md`

**PRP template & methodology**: `.github/instructions/PRP-README.md`

## AI-Bloat Prevention

- No giant drops: prefer small, incremental diffs
- No verbose/obvious comments, no placeholder TODOs
- Function complexity >15 → refactor; single function >120 LOC → refactor
- Minimize rename/reordering churn unless clarity win is undeniable

## Path-Specific Instructions

| Pattern | File | Purpose |
|---------|------|---------|
| `**/*.py`, `pyproject.toml` | [validation.instructions.md](instructions/validation.instructions.md) | Mandatory validation loop |
| `tests/**/*.py` | [tests.instructions.md](instructions/tests.instructions.md) | Pytest standards |
| GitHub CLI workflows | [gh-cli.instructions.md](instructions/gh-cli.instructions.md) | Non-interactive mode patterns |

## Prompt Library

Reusable templates in `.github/prompts/`:

- `prpPlanOnly.prompt.md` - Planning mode (no code changes until approved)
- `cleanCodeRefactor.prompt.md` - Rename/restructure without behavior changes
- `bugFixRapidResponse.prompt.md` - Quick surgical bug fixes
- `testHardening.prompt.md` - Improve test coverage/reliability
- `iterativeCodeSimplification.prompt.md` - Reduce LOC while maintaining behavior

**See [all prompts](prompts/)**

## Project-Specific Context

### Key Files
```yaml
- file: src/docs_html_screenshot/cli.py
  why: Main CLI entry point with Click commands

- file: Dockerfile
  why: Docker image using Playwright base image

- file: pyproject.toml
  why: Project configuration, dependencies, entry points
```

### Validation Commands
```bash
# Format and lint
uv run ruff format . && uv run ruff check --fix .

# Run tests
uv run pytest tests/ -v

# Test CLI locally
uv run docs-html-screenshot --help
uv run docs-html-screenshot --input <html_dir> --output <screenshot_dir>

# Build and test Docker image
docker build -t docs-html-screenshot .
docker run --rm docs-html-screenshot --help
```
