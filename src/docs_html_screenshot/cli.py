from __future__ import annotations

import asyncio
import os
import socket
import threading
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
from playwright.async_api import BrowserContext, async_playwright

DEFAULT_VIEWPORT_WIDTH = 1920
DEFAULT_VIEWPORT_HEIGHT = 1020
DEFAULT_DEVICE_SCALE_FACTOR = 2
DEFAULT_TIMEOUT_MS = 30000


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
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


@dataclass
class TaskResult:
    source: Path
    destination: Path
    errors: list[str]


def discover_html_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.html") if p.is_file()])


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = input_path.relative_to(input_root)
    target_name = relative.name + "-screenshot.png"
    return output_root.joinpath(relative.parent, target_name)


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


async def _process_file(
    context: BrowserContext,
    config: RunConfig,
    server_port: int,
    input_root: Path,
    output_root: Path,
    input_path: Path,
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

    output_path = output_path_for(input_path, input_root, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative = input_path.relative_to(input_root)
    url = build_url(server_port, relative)
    navigation_error: str | None = None

    try:
        await page.goto(url, wait_until="load", timeout=config.timeout_ms)
        await page.wait_for_load_state("load", timeout=config.timeout_ms)
        await page.screenshot(full_page=True, path=str(output_path))
    except Exception as exc:
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

    return TaskResult(source=input_path, destination=output_path, errors=errors)


async def run(config: RunConfig) -> int:
    html_files = discover_html_files(config.input_dir)
    if not html_files:
        click.echo("No HTML files found under input directory", err=True)
        return 0

    server, thread, port = _start_server(config.input_dir)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.headless)
            context = await browser.new_context(
                viewport={
                    "width": config.viewport_width,
                    "height": config.viewport_height,
                },
                device_scale_factor=config.device_scale_factor,
            )

            sem = asyncio.Semaphore(config.concurrency)

            async def runner(path: Path) -> TaskResult:
                async with sem:
                    return await _process_file(context, config, port, config.input_dir, config.output_dir, path)

            results = await asyncio.gather(*(runner(path) for path in html_files))
            await context.close()
            await browser.close()
    finally:
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
    required=True,
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
    "--headed/--headless",
    default=False,
    show_default=True,
    help="Run browser headed (default headless).",
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
def main(
    input_dir: Path,
    output_dir: Path,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: int,
    concurrency: int,
    timeout_ms: int,
    headed: bool,
    fail_on_http: bool,
    fail_on_console: bool,
    fail_on_pageerror: bool,
    fail_on_weberror: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
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
    )

    exit_code = asyncio.run(run(config))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
