# docs-html-screenshot

[![CI](https://github.com/pankaj28843/docs-html-screenshot/actions/workflows/ci.yml/badge.svg)](https://github.com/pankaj28843/docs-html-screenshot/actions/workflows/ci.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-docs--html--screenshot-blue)](https://ghcr.io/pankaj28843/docs-html-screenshot)
[![PyPI](https://img.shields.io/pypi/v/docs-html-screenshot)](https://pypi.org/project/docs-html-screenshot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Screenshot your MkDocs site, feed it to GitHub Copilot, get instant visual feedback.**

A CLI tool that captures full-page screenshots of static HTML files (MkDocs, Sphinx, Hugo output). Built for feeding documentation renders to LLMs with vision capabilities—so you can ask "what's wrong with this layout?" and get actionable answers.

**Audience:** Doc authors and developers who want fast, visual validation of static sites (MkDocs/Hugo/Sphinx) without manual browser work.

**Fastest path:**
- Have Docker? Use the one-liner below.
- Want to test URL mode? See the real example with https://github.com/pankaj28843/article-extractor in the URL section.

## What Is This?

**The problem:** You're writing MkDocs documentation, but you can't easily ask an LLM "does this page look right?" because LLMs need images, not HTML source.

**The solution:** This tool screenshots every page in your built site. Drop those PNGs into GitHub Copilot (or any vision-capable LLM) and ask:
- "Are there any rendering issues on this page?"
- "How could I improve this layout?"
- "Does the navigation look correct?"

**How it works:**
1. Build your static site (`mkdocs build`)
2. Run this tool on the output directory
3. Get PNG screenshots of every page
4. Feed to LLM for visual analysis

**Also useful for:** Visual regression testing, design QA, documentation archival.

## Who Is This For?

Developers who use MkDocs (or similar) and want to leverage LLM vision to review their documentation renders—without manually screenshotting each page.

### Prerequisites

You need **one of**:
- **Docker** (recommended) — zero setup, works anywhere
- **Python 3.10+** with `uv` or `pip`

### What You DON'T Need

- Playwright installation knowledge
- Browser driver management
- Complex configuration files

## Quick Start (Docker — Recommended)

Pre-built images are available on GitHub Container Registry. No `docker build` required.

```bash

# Minimal end-to-end
docker run --rm --init --ipc=host \
  -v "$PWD/site:/input:ro" \
  -v "$PWD/screenshots:/output" \
  ghcr.io/pankaj28843/docs-html-screenshot:latest \
  --input /input --output /output
# Pull the image (multi-arch: amd64/arm64)
docker pull ghcr.io/pankaj28843/docs-html-screenshot:latest
**Example output** (flat filenames for easy drag-and-drop):
# Generate screenshots from your static site
docker run --rm --init --ipc=host \
  -v "$PWD/site:/input:ro" \
  -v "$PWD/screenshots:/output" \
  ghcr.io/pankaj28843/docs-html-screenshot:latest \
  --input /input --output /output
```

**Example output** (flat filenames for easy drag-and-drop):
```
WROTE /output/index.html-screenshot.png
WROTE /output/getting-started__index.html-screenshot.png
WROTE /output/api__reference.html-screenshot.png
```

All screenshots are output as flat files using `__` as the path separator—no nested directories. This makes it trivial to select all PNGs and drag them into an LLM chat.

## Quick Start (Local Installation)

```bash
# Install with uv (recommended)
uv tool install docs-html-screenshot

# Or with pip
pip install docs-html-screenshot

# Install Playwright browsers (one-time)
playwright install chromium --with-deps

# Generate screenshots
docs-html-screenshot --input site --output screenshots
```

## Usage Reference

```
Usage: docs-html-screenshot [OPTIONS]

Options:
  --input DIRECTORY               Input directory containing HTML files.
                                  (optional; use --url/--urls-file for remote pages)
  --output DIRECTORY              Output directory for screenshots.
                                  [required]
  --url TEXT                      URL to screenshot (repeatable).
  --urls-file FILE                File containing URLs (one per line). Use '-'
                                  for stdin.
  --viewport-width INTEGER        Viewport width in pixels.  [default: 1920]
  --viewport-height INTEGER       Viewport height in pixels.  [default: 1020]
  --device-scale-factor INTEGER   Device scale factor for screenshots.
                                  [default: 2]
  --concurrency INTEGER           Number of concurrent pages.  [default: 10]
  --timeout-ms INTEGER            Navigation timeout in milliseconds.
                                  [default: 30000]
  --headed / --headless           Run browser headed (default headless).
                                  [default: headless]
  --fail-on-http / --allow-http-errors
                                  Fail build on HTTP status >= 400.  [default:
                                  fail-on-http]
  --fail-on-console / --allow-console-errors
                                  Fail build on console.error messages.
                                  [default: fail-on-console]
  --fail-on-pageerror / --allow-pageerror
                                  Fail build on page errors.  [default: fail-
                                  on-pageerror]
  --fail-on-weberror / --allow-weberror
                                  Fail build on web errors.  [default: fail-
                                  on-weberror]
  --help                          Show this message and exit.
```

### Configuration Examples

| Scenario | Command |
|----------|---------|
| **High-DPI (4K displays)** | `--viewport-width 2560 --viewport-height 1440 --device-scale-factor 2` |
| **Fast CI (lower quality)** | `--device-scale-factor 1 --concurrency 4` |
| **Permissive mode** | `--allow-http-errors --allow-console-errors --allow-pageerror` |
| **Debug rendering** | `--headed --concurrency 1 --timeout-ms 60000` |

## URL Mode (NEW)

Screenshot external pages without a local HTML directory:

```bash
# Single URL
docs-html-screenshot --url https://example.com --output ./screenshots

# Multiple URLs
docs-html-screenshot --url https://example.com --url https://another.com --output ./screenshots

# URLs from file (one per line)
docs-html-screenshot --urls-file urls.txt --output ./screenshots

# URLs from stdin
cat urls.txt | docs-html-screenshot --urls-file - --output ./screenshots

# Mixed mode: local directory + URLs
docs-html-screenshot --input ./docs --url https://example.com --output ./screenshots

# Real example: capture a GitHub repository page
docs-html-screenshot --url https://github.com/pankaj28843/article-extractor --output ./screenshots
# Expected file: ./screenshots/https__github.com__pankaj28843__article-extractor.png
```

**Example validation (Wikipedia):**

```bash
docs-html-screenshot --url https://en.wikipedia.org/wiki/Wikipedia --output ./screenshots
# Expected file: ./screenshots/https__en.wikipedia.org__wiki__Wikipedia.png
```

## Error Detection

By default, the tool fails (exit code 1) when any rendering error is detected. This is intentional—**catch broken documentation early**.

| Flag | What It Catches |
|------|-----------------|
| `--fail-on-http` | HTTP responses with status ≥ 400 (missing images, broken links) |
| `--fail-on-console` | `console.error()` calls (JavaScript errors logged to console) |
| `--fail-on-pageerror` | Uncaught JavaScript exceptions |
| `--fail-on-weberror` | Network failures, CORS errors, failed resource loads |

**Screenshots are still generated** even when errors are detected—this helps with debugging.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All pages rendered successfully |
| `1` | One or more pages had errors (screenshots still generated) |

## Docker Usage

### Using Pre-built Images (Recommended)

Images are automatically built and pushed to GHCR on every push to `main`. **Always use `:latest`** for the most up-to-date version:

```bash
# Always use :latest (tracks main branch)
docker pull ghcr.io/pankaj28843/docs-html-screenshot:latest
```

**Available tags:**

| Tag | When Updated |
|-----|-------------|
| `:latest` | Every push to `main` branch |
| `:v0.1.0` | Semantic version releases |
| `:sha-abc1234` | Specific commit SHA |

### Volume Mounting

```bash
docker run --rm --init --ipc=host \
  -v "/path/to/html:/input:ro" \      # Read-only input
  -v "/path/to/output:/output" \       # Writable output
  ghcr.io/pankaj28843/docs-html-screenshot:latest \
  --input /input --output /output
```

**Important Docker flags** (per [Playwright Docker docs](https://playwright.dev/python/docs/docker)):
- `--init` — Proper process cleanup, avoids zombie processes
- `--ipc=host` — Prevents Chromium OOM crashes
- `--rm` — Clean up container after exit

### GitHub Actions Integration

```yaml
jobs:
  screenshots:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build documentation
        run: mkdocs build  # or your static site generator
      
      - name: Generate screenshots
        run: |
          docker run --rm --init --ipc=host \
            -v "${{ github.workspace }}/site:/input:ro" \
            -v "${{ github.workspace }}/screenshots:/output" \
            ghcr.io/pankaj28843/docs-html-screenshot:latest \
            --input /input --output /output
      
      - name: Upload screenshots
        uses: actions/upload-artifact@v4
        with:
          name: documentation-screenshots
          path: screenshots/
```

### Building Custom Images

Only needed if you want to customize the base image:

```bash
docker build -t my-screenshot-tool .

docker run --rm --init --ipc=host \
  -v "$PWD/site:/input:ro" \
  -v "$PWD/screenshots:/output" \
  my-screenshot-tool --input /input --output /output
```

## Design Decisions

### Why 1920×1020 viewport?

- **1920px width**: Common desktop width, captures most responsive layouts without horizontal scroll
- **1020px height**: Slightly less than 1080p to account for browser chrome in visual comparisons
- Full-page screenshots capture content beyond the viewport, so height mainly affects above-the-fold rendering

### Why device scale factor 2?

- Produces **retina-quality screenshots** (3840×2040 effective resolution)
- Text remains crisp when zooming into screenshots
- Modern displays are increasingly high-DPI; scale factor 1 looks dated

### Why fail by default on errors?

- **Catch broken documentation early** in CI pipelines
- Missing images, broken links, JS errors are real bugs—don't silently ignore them
- Screenshots are still generated for debugging, just exit code ≠ 0

### Why concurrency = CPU count?

- **Balanced resource usage**: Each Chromium page consumes memory
- 10 concurrent pages is reasonable for most machines
- Reduce with `--concurrency 2-4` if running in memory-constrained containers

### Why embedded HTTP server?

- File protocol (`file://`) has CORS restrictions that break many sites
- Embedded server serves files on localhost, matching production behavior
- Zero configuration—server starts/stops automatically

### Why flat output filenames?

- **Easy drag-and-drop**: Select all PNGs at once without navigating nested folders
- **LLM-friendly**: Drop entire folder into Claude, ChatGPT, or Copilot Vision
- **Path preserved**: Original path is encoded in filename (`docs/api/index.html` → `docs__api__index.html-screenshot.png`)

## Troubleshooting

### Blurry screenshots

**Cause**: Device scale factor too low.

```bash
--device-scale-factor 2  # or higher
```

### Out of memory (OOM) in Docker

**Cause**: Too many concurrent pages.

```bash
--concurrency 2  # Reduce parallelism
```

Or increase Docker memory limit:
```bash
docker run --memory=4g ...
```

### Timeout errors

**Cause**: Pages take too long to load.

```bash
--timeout-ms 60000  # Increase to 60 seconds
```

### Permission denied on output directory

**Cause**: Docker volume mount permissions.

```bash
# Ensure output directory exists and is writable
mkdir -p screenshots
chmod 777 screenshots  # Or use proper user mapping
```

### Blank/white screenshots

**Cause**: Page requires JavaScript execution time.

The tool waits for `load` event, but some SPAs need more time. Consider:
- Ensuring your static site is truly static (pre-rendered HTML)
- Increasing timeout with `--timeout-ms`

### "Executable doesn't exist" error (local install)

**Cause**: Playwright browsers not installed.

```bash
playwright install chromium --with-deps
```

## Development

```bash
# Clone and install
git clone https://github.com/pankaj28843/docs-html-screenshot.git
cd docs-html-screenshot
uv sync --extra dev

# Install Playwright browsers
uv run playwright install chromium --with-deps

# Run tests
uv run pytest tests/ -v

# Format and lint
uv run ruff format . && uv run ruff check --fix .

# Test CLI
uv run docs-html-screenshot --help
```

## License

MIT
