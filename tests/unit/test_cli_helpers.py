from pathlib import Path

import pytest
from click.testing import CliRunner

from docs_html_screenshot import cli


@pytest.mark.unit
def test_output_path_flat_filename(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    (input_root / "a" / "b").mkdir(parents=True)
    source = input_root / "a" / "b" / "index.html"
    source.write_text("<html></html>")

    target = cli.output_path_for(source, input_root, output_root)

    assert target == output_root / "a__b__index.html-screenshot.png"


@pytest.mark.unit
def test_output_path_root_level_file(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    source = input_root / "index.html"
    source.write_text("<html></html>")

    target = cli.output_path_for(source, input_root, output_root)

    assert target == output_root / "index.html-screenshot.png"


@pytest.mark.unit
def test_url_to_output_path_basic(tmp_path):
    output_root = tmp_path / "output"

    target = cli.url_to_output_path("https://example.com/docs/index.html", output_root)

    assert target.name == "https__example.com__docs__index.html.png"
    assert target.parent == output_root


@pytest.mark.unit
def test_url_to_output_path_truncates_long_urls(tmp_path):
    output_root = tmp_path / "output"
    long_url = "https://example.com/" + "path/" * 60 + "?q=1"

    target = cli.url_to_output_path(long_url, output_root)

    assert target.name.endswith(".png")
    assert len(target.name) < 240  # hashed suffix applied


@pytest.mark.unit
def test_discover_html_files_filters_non_html(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    html_file = input_root / "page.html"
    other_file = input_root / "notes.txt"
    html_file.write_text("<html></html>")
    other_file.write_text("text")

    found = cli.discover_html_files(input_root)

    assert found == [html_file]


@pytest.mark.unit
def test_discover_html_files_empty_directory(tmp_path):
    input_root = tmp_path / "empty"
    input_root.mkdir()

    found = cli.discover_html_files(input_root)

    assert found == []


@pytest.mark.unit
def test_discover_html_files_finds_nested_files(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "sub" / "deep").mkdir(parents=True)
    root_file = input_root / "index.html"
    sub_file = input_root / "sub" / "page.html"
    deep_file = input_root / "sub" / "deep" / "nested.html"
    root_file.write_text("<html></html>")
    sub_file.write_text("<html></html>")
    deep_file.write_text("<html></html>")

    found = cli.discover_html_files(input_root)

    assert len(found) == 3
    assert root_file in found
    assert sub_file in found
    assert deep_file in found


@pytest.mark.unit
def test_discover_html_files_returns_sorted(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "z.html").write_text("<html></html>")
    (input_root / "a.html").write_text("<html></html>")
    (input_root / "m.html").write_text("<html></html>")

    found = cli.discover_html_files(input_root)

    assert found == sorted(found)
    assert found[0].name == "a.html"
    assert found[-1].name == "z.html"


@pytest.mark.unit
def test_build_url_formats_correctly():
    result = cli.build_url(8080, Path("docs/index.html"))

    assert result == "http://127.0.0.1:8080/docs/index.html"


@pytest.mark.unit
def test_build_url_root_path():
    result = cli.build_url(3000, Path("index.html"))

    assert result == "http://127.0.0.1:3000/index.html"


@pytest.mark.unit
def test_pick_free_port_returns_valid_port():
    port = cli._pick_free_port()

    assert isinstance(port, int)
    assert 1024 <= port <= 65535


@pytest.mark.unit
def test_run_config_defaults():
    config = cli.RunConfig(
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        urls=[],
    )

    assert config.viewport_width == cli.DEFAULT_VIEWPORT_WIDTH
    assert config.viewport_height == cli.DEFAULT_VIEWPORT_HEIGHT
    assert config.device_scale_factor == cli.DEFAULT_DEVICE_SCALE_FACTOR
    assert config.timeout_ms == cli.DEFAULT_TIMEOUT_MS
    assert config.headless is True
    assert config.fail_on_http is True
    assert config.fail_on_console is True
    assert config.fail_on_pageerror is True
    assert config.fail_on_weberror is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_target_uses_config_timeout(tmp_path):
    class FakePage:
        def __init__(self):
            self.screenshot_kwargs = None
            self.goto_kwargs = None

        def on(self, *_args, **_kwargs):
            return None

        def off(self, *_args, **_kwargs):
            return None

        async def goto(self, *_args, **kwargs):
            self.goto_kwargs = kwargs

        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

        async def screenshot(self, **kwargs):
            self.screenshot_kwargs = kwargs
            return b"img"

        async def close(self):
            return None

    class FakeContext:
        def __init__(self):
            self.page = FakePage()

        async def new_page(self):
            return self.page

        def on(self, *_args, **_kwargs):
            return None

        def off(self, *_args, **_kwargs):
            return None

    context = FakeContext()
    config = cli.RunConfig(input_dir=None, output_dir=tmp_path, urls=[], timeout_ms=12345)

    result = await cli._capture_target(
        context,
        config,
        "https://example.com",
        tmp_path / "out.png",
        "example",
    )

    assert result.errors == []
    assert context.page.screenshot_kwargs == {"full_page": True}
    assert context.page.goto_kwargs is not None
    assert context.page.goto_kwargs["wait_until"] == "domcontentloaded"
    assert (tmp_path / "out.png").read_bytes() == b"img"


@pytest.mark.unit
def test_apply_timeouts_calls_context_methods():
    class FakeContext:
        def __init__(self):
            self.default_nav_timeout = None

        def set_default_navigation_timeout(self, value):
            self.default_nav_timeout = value

    ctx = FakeContext()
    config = cli.RunConfig(input_dir=None, output_dir=Path("/out"), urls=[], timeout_ms=4567)

    cli._apply_timeouts(ctx, config)

    assert ctx.default_nav_timeout == 4567


@pytest.mark.unit
def test_task_result_stores_errors():
    result = cli.TaskResult(
        source=Path("/input/page.html"),
        destination=Path("/output/page.html-screenshot.png"),
        errors=["HTTP 404: /missing.js"],
    )

    assert result.source == Path("/input/page.html")
    assert result.destination == Path("/output/page.html-screenshot.png")
    assert len(result.errors) == 1
    assert "HTTP 404" in result.errors[0]


@pytest.mark.unit
def test_click_help_runs_without_execution():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "docs-html-screenshot" in result.output


@pytest.mark.unit
def test_cli_requires_output_option():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("input").mkdir()
        result = runner.invoke(cli.main, ["--input", "input"])

    assert result.exit_code != 0
    assert "Missing option '--output'" in result.output


@pytest.mark.unit
def test_cli_validates_input_exists():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--input", "/nonexistent", "--output", "/tmp/out"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


@pytest.mark.unit
def test_load_urls_combines_cli_and_file(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://a.example.com\n# comment\nhttps://b.example.com\nhttps://a.example.com\n")

    urls = cli.load_urls(("https://c.example.com",), str(url_file))

    assert urls == ["https://c.example.com", "https://a.example.com", "https://b.example.com"]


@pytest.mark.unit
def test_cli_accepts_url_without_input(monkeypatch):
    called: dict[str, list[str]] = {}

    async def fake_run(config: cli.RunConfig) -> int:  # type: ignore[override]
        called["urls"] = config.urls
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli.main,
            ["--url", "https://example.com", "--output", "out"],
        )

    assert result.exit_code == 0
    assert called["urls"] == ["https://example.com"]


@pytest.mark.unit
def test_cli_requires_input_or_urls():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--output", "/tmp/out"])

    assert result.exit_code != 0
    assert "Provide --input or --url/--urls-file." in result.output


@pytest.mark.unit
def test_start_stop_server(tmp_path):
    input_dir = tmp_path / "html"
    input_dir.mkdir()
    (input_dir / "index.html").write_text("<html><body>Test</body></html>")

    server, thread, port = cli._start_server(input_dir)

    assert thread.is_alive()
    assert 1024 <= port <= 65535

    cli._stop_server(server, thread)

    assert not thread.is_alive()
