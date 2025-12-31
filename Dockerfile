FROM mcr.microsoft.com/playwright/python:v1.56.0-noble

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
COPY src/ ./src/

ENTRYPOINT ["uv", "run", "docs-html-screenshot"]
