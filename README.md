# docs-html-screenshot

**Generate full-page screenshots of static HTML files using Playwright**

A CLI tool that scans a directory of HTML files (e.g., MkDocs `site/` output), serves them locally, and captures full-page screenshots with error detection.

## Features

- 📸 Full-page screenshots at configurable viewport/resolution
- 🔍 Detects rendering errors: console.error, page errors, HTTP failures
- ⚡ Concurrent processing for speed
- 🐳 Docker-ready with Playwright image

## Installation

```bash
# Install with uv
uv tool install docs-html-screenshot

# Or install from source
uv tool install --refresh --force --editable .
```

## Quick Start

```bash
# Build your static site (e.g., MkDocs)
mkdocs build

# Generate screenshots
docs-html-screenshot --input site --output screenshots
```

## Usage

```bash
docs-html-screenshot --input <HTML_DIR> --output <SCREENSHOT_DIR> [OPTIONS]

Options:
  --viewport-width INT      Viewport width (default: 1920)
  --viewport-height INT     Viewport height (default: 1020)
  --device-scale-factor INT Scale factor for retina (default: 2)
  --concurrency INT         Parallel pages (default: CPU count)
  --timeout-ms INT          Navigation timeout (default: 30000)
  --headed/--headless       Run browser visibly
  --fail-on-http/--allow-http-errors      Fail on HTTP >= 400
  --fail-on-console/--allow-console-errors Fail on console.error
  --fail-on-pageerror/--allow-pageerror   Fail on page errors
  --fail-on-weberror/--allow-weberror     Fail on web errors
```

## Exit Codes

- `0`: All pages rendered successfully
- `1`: One or more pages had errors (screenshots still generated for debugging)

## Docker

```bash
# Build image
docker build -t docs-html-screenshot .

# Run
docker run --rm --init --ipc=host \
  -v "$PWD/site:/input:ro" \
  -v "$PWD/screenshots:/output" \
  docs-html-screenshot --input /input --output /output
```

## License

MIT
