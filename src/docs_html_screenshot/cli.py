from __future__ import annotations

import asyncio
import fileinput
import hashlib
import os
import socket
import threading
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import click
from fake_useragent import UserAgent
from playwright.async_api import BrowserContext, async_playwright

DEFAULT_VIEWPORT_WIDTH = 1920
DEFAULT_VIEWPORT_HEIGHT = 1020
DEFAULT_DEVICE_SCALE_FACTOR = 2
DEFAULT_TIMEOUT_MS = 30000
DEFAULT_USER_INTERACTION_TIMEOUT = 60
DEFAULT_STORAGE_STATE_PATH = Path.home() / ".dhsu" / "storage_state.json"
LOCALHOST_BYPASS_ENTRIES = {"127.0.0.1", "localhost"}
DOM_STABILITY_MAX_CHECKS = 20
DOM_STABILITY_INTERVAL = 0.5


async def _with_timeout(coro, timeout_seconds: float):
    """Run coroutine with a hard timeout and swallow cancelled-task errors."""
    task = asyncio.create_task(coro)
    try:
        return await asyncio.wait_for(task, timeout_seconds)
    except Exception:
        task.cancel()
        with suppress(Exception):
            await task
        raise


@dataclass
class RunConfig:
    input_dir: Path | None
    output_dir: Path
    urls: list[str] = field(default_factory=list)
    viewport_width: int = DEFAULT_VIEWPORT_WIDTH
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT
    device_scale_factor: int = DEFAULT_DEVICE_SCALE_FACTOR
    concurrency: int = max(1, os.cpu_count() or 1)
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    headless: bool = True
    fail_on_http: bool = True
    fail_on_console: bool = True
    fail_on_pageerror: bool = True
    fail_on_weberror: bool = True
    user_agent: str | None = None
    proxy: ProxySettings | None = None
    storage_state_path: Path | None = None
    user_interaction_timeout: int = DEFAULT_USER_INTERACTION_TIMEOUT


@dataclass
class ProxySettings:
    server: str
    username: str | None = None
    password: str | None = None
    bypass: str | None = None

    def to_playwright_options(self) -> dict[str, str]:
        payload: dict[str, str] = {"server": self.server}
        if self.username:
            payload["username"] = self.username
        if self.password:
            payload["password"] = self.password
        if self.bypass:
            payload["bypass"] = self.bypass
        return payload


@dataclass
class TaskResult:
    source: str | Path
    destination: Path
    errors: list[str]


def _resolve_storage_state_path(persist: bool, storage_path: Path | None) -> Path | None:
    if not persist:
        return None
    if storage_path is None:
        return None
    return storage_path.expanduser()


def _build_random_user_agent() -> str | None:
    try:
        return UserAgent().random
    except Exception as exc:  # pragma: no cover - best effort logging
        click.echo(f"Failed to generate random user agent: {exc}", err=True)
        return None


def _determine_user_agent(explicit: str | None, randomize: bool) -> str | None:
    if explicit:
        return explicit
    if randomize:
        return _build_random_user_agent()
    return None


def _normalize_proxy_url(raw: str) -> tuple[str, str | None, str | None]:
    value = raw.strip()
    if not value:
        raise click.ClickException("Proxy server value cannot be empty.")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if not parsed.hostname:
        raise click.ClickException(f"Invalid proxy server: {raw}")
    scheme = parsed.scheme or "http"
    server = f"{scheme}://{parsed.hostname}"
    if parsed.port:
        server = f"{server}:{parsed.port}"
    return server, parsed.username, parsed.password


def _format_bypass_entries(values: list[str]) -> str | None:
    cleaned = {entry.strip() for entry in values if entry and entry.strip()}
    cleaned.update(LOCALHOST_BYPASS_ENTRIES)
    return ",".join(sorted(cleaned)) if cleaned else None


def _select_proxy_value(cli_value: str | None, env: Mapping[str, str]) -> str | None:
    if cli_value:
        return cli_value
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        if env.get(key):
            return env[key]
    return None


def _build_proxy_settings(
    proxy_value: str | None,
    proxy_bypass: str | None,
    env: Mapping[str, str],
) -> ProxySettings | None:
    selected = _select_proxy_value(proxy_value, env)
    if not selected:
        return None
    server, username, password = _normalize_proxy_url(selected)
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy")
    bypass = _format_bypass_entries([proxy_bypass or "", no_proxy or ""])
    return ProxySettings(server=server, username=username, password=password, bypass=bypass)


async def _maybe_wait_for_user_interaction(config: RunConfig, target_url: str) -> None:
    if config.headless or config.user_interaction_timeout <= 0:
        return
    click.echo(
        f"Waiting {config.user_interaction_timeout}s for manual interaction on {target_url}. "
        "Use --user-interaction-timeout 0 to skip.",
        err=True,
    )
    await asyncio.sleep(config.user_interaction_timeout)


async def _wait_for_dom_stability(
    page,
    max_checks: int = DOM_STABILITY_MAX_CHECKS,
    interval_seconds: float = DOM_STABILITY_INTERVAL,
) -> None:
    previous = None
    for _ in range(max_checks):
        await asyncio.sleep(interval_seconds)
        try:
            current = await page.content()
        except Exception as exc:  # pragma: no cover - visibility only
            click.echo(f"Failed to inspect page content during stability check: {exc}", err=True)
            return
        if current == previous:
            break
        previous = current


def discover_html_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.html") if p.is_file()])


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = input_path.relative_to(input_root)
    # Flatten directory structure: use __ as separator for flat output
    flat_name = str(relative).replace("/", "__").replace("\\", "__")
    target_name = flat_name + "-screenshot.png"
    return output_root / target_name


def url_to_output_path(url: str, output_root: Path) -> Path:
    sanitized = url.replace("://", "__").replace("/", "__").replace(":", "__").replace("?", "__q__")
    if len(sanitized) > 200:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
        sanitized = f"{sanitized[:150]}__{digest}"
    return output_root / f"{sanitized}.png"


def load_urls(url_flags: tuple[str, ...], urls_file: str | None) -> list[str]:
    urls = [item for item in url_flags if item]
    if urls_file:
        try:
            with fileinput.input(files=(urls_file,)) as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        urls.append(stripped)
        except OSError as exc:
            raise click.ClickException(f"Failed to read URLs from {urls_file}: {exc}") from exc

    # Preserve order while dropping duplicates
    seen: dict[str, None] = {}
    for item in urls:
        seen.setdefault(item, None)
    return list(seen.keys())


def build_url(port: int, relative_path: Path) -> str:
    return f"http://127.0.0.1:{port}/{relative_path.as_posix()}"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_server(root: Path) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    port = _pick_free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, port


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


async def _capture_target(
    context: BrowserContext,
    config: RunConfig,
    target_url: str,
    output_path: Path,
    source_label: str | Path,
) -> TaskResult:
    page = await context.new_page()
    console_errors: list[str] = []
    page_errors: list[str] = []
    http_errors: list[str] = []
    web_errors: list[str] = []

    def on_console(msg) -> None:
        if msg.type == "error":
            console_errors.append(msg.text)

    def on_page_error(exc) -> None:
        page_errors.append(str(exc))

    def on_response(resp) -> None:
        status_attr = getattr(resp, "status", None)
        status_value = status_attr() if callable(status_attr) else status_attr
        if status_value is not None and status_value >= 400:
            http_errors.append(f"HTTP {status_value}: {resp.url}")

    def on_weberror(err) -> None:
        if getattr(err, "page", None) in (None, page):
            web_errors.append(str(getattr(err, "error", err)))

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("response", on_response)
    context.on("weberror", on_weberror)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    navigation_error: str | None = None
    timeout_seconds = max(0.001, config.timeout_ms / 1000)

    try:
        await _with_timeout(page.goto(target_url, wait_until="domcontentloaded"), timeout_seconds)
        await _maybe_wait_for_user_interaction(config, target_url)
        await _with_timeout(_wait_for_dom_stability(page), timeout_seconds)
        screenshot_bytes = await _with_timeout(page.screenshot(full_page=True), timeout_seconds)
        output_path.write_bytes(screenshot_bytes)
    except BaseException as exc:  # CancelledError may not derive from Exception in all py versions
        if isinstance(exc, asyncio.CancelledError):
            navigation_error = str(exc)
        else:
            navigation_error = str(exc)
    finally:
        with suppress(Exception):
            page.off("console", on_console)
            page.off("pageerror", on_page_error)
            page.off("response", on_response)
        with suppress(Exception):
            context.off("weberror", on_weberror)
        await page.close()

    errors: list[str] = []
    if config.fail_on_console:
        errors.extend(console_errors)
    if config.fail_on_pageerror:
        errors.extend(page_errors)
    if config.fail_on_http:
        errors.extend(http_errors)
    if config.fail_on_weberror:
        errors.extend(web_errors)
    if navigation_error:
        errors.append(f"navigation_error: {navigation_error}")
    if not output_path.exists():
        errors.append(f"screenshot_missing: {output_path}")

    return TaskResult(source=source_label, destination=output_path, errors=errors)


def _apply_timeouts(context: BrowserContext, config: RunConfig) -> None:
    """Apply per-context timeouts so every operation respects config.timeout_ms."""
    with suppress(Exception):
        context.set_default_navigation_timeout(config.timeout_ms)


async def run(config: RunConfig) -> int:  # noqa: PLR0912 - orchestration requires several branches
    html_files = discover_html_files(config.input_dir) if config.input_dir else []
    url_targets = config.urls

    if not html_files and not url_targets:
        click.echo("No inputs provided. Use --input or --url/--urls-file.", err=True)
        return 1

    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    port: int | None = None
    results: list[TaskResult] = []

    if html_files and config.input_dir:
        server, thread, port = _start_server(config.input_dir)

    try:
        async with async_playwright() as p:
            proxy_options = config.proxy.to_playwright_options() if config.proxy else None
            browser = await p.chromium.launch(headless=config.headless, proxy=proxy_options)
            context: BrowserContext | None = None
            try:
                context_kwargs: dict[str, object] = {
                    "viewport": {
                        "width": config.viewport_width,
                        "height": config.viewport_height,
                    },
                    "device_scale_factor": config.device_scale_factor,
                }
                if config.user_agent:
                    context_kwargs["user_agent"] = config.user_agent
                if config.storage_state_path and config.storage_state_path.exists():
                    context_kwargs["storage_state"] = str(config.storage_state_path)

                context = await browser.new_context(**context_kwargs)
                _apply_timeouts(context, config)

                effective_concurrency = 1 if not config.headless else max(1, config.concurrency)
                sem = asyncio.Semaphore(effective_concurrency)

                async def runner(source_label: str | Path, target_url: str, output_path: Path) -> TaskResult:
                    async with sem:
                        return await _capture_target(context, config, target_url, output_path, source_label)

                jobs: list[asyncio.Future[TaskResult]] = []
                if html_files and port is not None and config.input_dir:
                    for path in html_files:
                        target_url = build_url(port, path.relative_to(config.input_dir))
                        output_path = output_path_for(path, config.input_dir, config.output_dir)
                        jobs.append(asyncio.ensure_future(runner(path, target_url, output_path)))

                for target in url_targets:
                    output_path = url_to_output_path(target, config.output_dir)
                    jobs.append(asyncio.ensure_future(runner(target, target, output_path)))

                results = await asyncio.gather(*jobs)
            finally:
                if context:
                    if config.storage_state_path:
                        try:
                            config.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
                            await context.storage_state(path=str(config.storage_state_path))
                        except Exception as exc:  # pragma: no cover - logging only
                            click.echo(f"Failed to persist storage state: {exc}", err=True)
                    await context.close()
                await browser.close()
    finally:
        if server and thread:
            _stop_server(server, thread)

    failures = [r for r in results if r.errors]
    for item in failures:
        click.echo(f"FAIL {item.source}: {item.errors}", err=True)
    for item in results:
        click.echo(f"WROTE {item.destination}")

    return 1 if failures else 0


@click.command(name="docs-html-screenshot")
@click.option(
    "--input",
    "input_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=False,
    help="Input directory containing HTML files.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Output directory for screenshots.",
)
@click.option(
    "--url",
    "urls",
    multiple=True,
    help="URL to screenshot (repeatable).",
)
@click.option(
    "--urls-file",
    type=click.Path(path_type=Path, dir_okay=False, allow_dash=True),
    help="File containing URLs (one per line). Use '-' for stdin.",
)
@click.option(
    "--viewport-width",
    default=DEFAULT_VIEWPORT_WIDTH,
    show_default=True,
    help="Viewport width in pixels.",
)
@click.option(
    "--viewport-height",
    default=DEFAULT_VIEWPORT_HEIGHT,
    show_default=True,
    help="Viewport height in pixels.",
)
@click.option(
    "--device-scale-factor",
    default=DEFAULT_DEVICE_SCALE_FACTOR,
    show_default=True,
    help="Device scale factor for screenshots.",
)
@click.option(
    "--concurrency",
    default=max(1, os.cpu_count() or 1),
    show_default=True,
    help="Number of concurrent pages.",
)
@click.option(
    "--timeout-ms",
    default=DEFAULT_TIMEOUT_MS,
    show_default=True,
    help="Navigation timeout in milliseconds.",
)
@click.option(
    "--user-agent",
    help="Explicit user agent string to use.",
)
@click.option(
    "--random-user-agent/--no-random-user-agent",
    default=True,
    show_default=True,
    help="Generate a realistic desktop user agent when one is not specified.",
)
@click.option(
    "--proxy-server",
    help="Proxy server (e.g. http://user:pass@host:3128).",
)
@click.option(
    "--proxy-bypass",
    help="Comma-separated hostnames that should bypass the proxy.",
)
@click.option(
    "--user-interaction-timeout",
    default=DEFAULT_USER_INTERACTION_TIMEOUT,
    show_default=True,
    help="Seconds to pause in headed mode before capturing screenshots.",
)
@click.option(
    "--headed/--headless",
    default=False,
    show_default=True,
    help=(
        "Run browser headed (default headless). "
        "Headed runs wait for --user-interaction-timeout seconds before capturing."
    ),
)
@click.option(
    "--fail-on-http/--allow-http-errors",
    default=True,
    show_default=True,
    help="Fail build on HTTP status >= 400.",
)
@click.option(
    "--fail-on-console/--allow-console-errors",
    default=True,
    show_default=True,
    help="Fail build on console.error messages.",
)
@click.option(
    "--fail-on-pageerror/--allow-pageerror",
    default=True,
    show_default=True,
    help="Fail build on page errors.",
)
@click.option(
    "--fail-on-weberror/--allow-weberror",
    default=True,
    show_default=True,
    help="Fail build on web errors.",
)
@click.option(
    "--storage-state",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_STORAGE_STATE_PATH,
    show_default=True,
    help="Path for persisted Playwright storage state.",
)
@click.option(
    "--persist-storage/--no-persist-storage",
    default=True,
    show_default=True,
    help="Persist authenticated storage between runs.",
)
def main(
    input_dir: Path | None,
    output_dir: Path,
    urls: tuple[str, ...],
    urls_file: Path | None,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: int,
    concurrency: int,
    timeout_ms: int,
    user_agent: str | None,
    random_user_agent: bool,
    proxy_server: str | None,
    proxy_bypass: str | None,
    user_interaction_timeout: int,
    headed: bool,
    fail_on_http: bool,
    fail_on_console: bool,
    fail_on_pageerror: bool,
    fail_on_weberror: bool,
    storage_state: Path,
    persist_storage: bool,
) -> None:
    ctx = click.get_current_context()
    urls_list = load_urls(urls, str(urls_file) if urls_file else None)
    if input_dir is None and not urls_list:
        raise click.UsageError("Provide --input or --url/--urls-file.", ctx=ctx)

    output_dir.mkdir(parents=True, exist_ok=True)

    storage_state_path = _resolve_storage_state_path(persist_storage, storage_state)
    proxy_settings = _build_proxy_settings(proxy_server, proxy_bypass, os.environ)
    resolved_user_agent = _determine_user_agent(user_agent, random_user_agent)

    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        urls=urls_list,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        device_scale_factor=device_scale_factor,
        concurrency=max(1, concurrency),
        timeout_ms=timeout_ms,
        headless=not headed,
        fail_on_http=fail_on_http,
        fail_on_console=fail_on_console,
        fail_on_pageerror=fail_on_pageerror,
        fail_on_weberror=fail_on_weberror,
        user_agent=resolved_user_agent,
        proxy=proxy_settings,
        storage_state_path=storage_state_path,
        user_interaction_timeout=user_interaction_timeout,
    )

    exit_code = asyncio.run(run(config))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
