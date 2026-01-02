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
    assert config.user_agent is None
    assert config.proxy is None
    assert config.storage_state_path is None
    assert config.user_interaction_timeout == cli.DEFAULT_USER_INTERACTION_TIMEOUT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capture_target_uses_config_timeout(tmp_path):
    class FakePage:
        def __init__(self):
            self.screenshot_kwargs = None
            self.goto_kwargs = None
            self.events = []

        def on(self, *_args, **_kwargs):
            return None

        def off(self, *_args, **_kwargs):
            return None

        async def goto(self, *_args, **kwargs):
            self.goto_kwargs = kwargs

        async def wait_for_load_state(self, *_args, **_kwargs):
            return None

        async def content(self):
            return "stable"

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
@pytest.mark.asyncio
async def test_capture_target_waits_for_dom_stability_before_screenshot(monkeypatch, tmp_path):
    events: list[str] = []

    async def fake_wait(page, *args, **kwargs):
        events.append("dom-stable")
        assert page is fake_context.page

    monkeypatch.setattr(cli, "_wait_for_dom_stability", fake_wait)

    class FakePage:
        def __init__(self):
            self.screenshot_taken = False
            self.goto_kwargs = None

        def on(self, *_args, **_kwargs):
            return None

        def off(self, *_args, **_kwargs):
            return None

        async def goto(self, *_args, **kwargs):
            self.goto_kwargs = kwargs

        async def screenshot(self, **_kwargs):
            events.append("screenshot")
            self.screenshot_taken = True
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

    fake_context = FakeContext()
    config = cli.RunConfig(input_dir=None, output_dir=tmp_path, urls=[], timeout_ms=1000)

    await cli._capture_target(
        fake_context,
        config,
        "https://example.com",
        tmp_path / "out.png",
        "example",
    )

    assert events == ["dom-stable", "screenshot"]
    assert fake_context.page.screenshot_taken is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_for_dom_stability_breaks_when_content_stops_changing():
    class FakePage:
        def __init__(self):
            self.values = ["state-1", "state-2", "state-2", "state-2"]
            self.index = 0
            self.calls = 0

        async def content(self):
            self.calls += 1
            value = self.values[self.index]
            if self.index < len(self.values) - 1:
                self.index += 1
            return value

    page = FakePage()

    await cli._wait_for_dom_stability(page, max_checks=10, interval_seconds=0)

    assert page.calls == 3  # stopped after detecting stability


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_for_dom_stability_stops_after_max_checks():
    class FlappingPage:
        def __init__(self):
            self.calls = 0

        async def content(self):
            self.calls += 1
            return f"state-{self.calls}"

    page = FlappingPage()

    await cli._wait_for_dom_stability(page, max_checks=5, interval_seconds=0)

    assert page.calls == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_for_dom_stability_logs_failure(monkeypatch):
    class FailingPage:
        async def content(self):  # pragma: no cover - exercised via helper
            raise RuntimeError("boom")

    captured: dict[str, str] = {}

    def fake_echo(message, err=False):
        captured["msg"] = message
        captured["err"] = str(err)

    monkeypatch.setattr(cli.click, "echo", fake_echo)

    await cli._wait_for_dom_stability(FailingPage(), max_checks=1, interval_seconds=0)

    assert "Failed to inspect page content" in captured.get("msg", "")


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


def test_determine_user_agent_prefers_explicit(monkeypatch):
    monkeypatch.setattr(cli, "_build_random_user_agent", lambda: "random-agent")

    assert cli._determine_user_agent("custom", True) == "custom"
    assert cli._determine_user_agent(None, True) == "random-agent"
    assert cli._determine_user_agent(None, False) is None


@pytest.mark.unit
def test_build_proxy_settings_uses_env_when_cli_missing():
    env = {"HTTPS_PROXY": "http://proxy.local:9000", "NO_PROXY": "internal.local"}
    settings = cli._build_proxy_settings(None, None, env)

    assert settings is not None
    data = settings.to_playwright_options()
    assert data["server"] == "http://proxy.local:9000"
    assert "internal.local" in data["bypass"]
    assert "localhost" in data["bypass"]


@pytest.mark.unit
def test_resolve_storage_state_path_respects_toggle(tmp_path):
    target = tmp_path / "storage.json"
    assert cli._resolve_storage_state_path(True, target) == target
    assert cli._resolve_storage_state_path(False, target) is None


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
    called: dict[str, cli.RunConfig] = {}

    async def fake_run(config: cli.RunConfig) -> int:  # type: ignore[override]
        called["config"] = config
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "_determine_user_agent", lambda *args, **_kwargs: "test-agent")

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli.main,
            ["--url", "https://example.com", "--output", "out"],
        )

    assert result.exit_code == 0
    config = called["config"]
    assert config.urls == ["https://example.com"]
    assert config.user_agent == "test-agent"
    assert config.proxy is None
    assert config.storage_state_path == cli.DEFAULT_STORAGE_STATE_PATH
    assert config.user_interaction_timeout == cli.DEFAULT_USER_INTERACTION_TIMEOUT


@pytest.mark.unit
def test_cli_requires_input_or_urls():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--output", "/tmp/out"])

    assert result.exit_code != 0
    assert "Provide --input or --url/--urls-file." in result.output


@pytest.mark.unit
def test_cli_allows_proxy_and_storage_overrides(monkeypatch):
    captured: dict[str, cli.RunConfig] = {}

    async def fake_run(config: cli.RunConfig) -> int:  # type: ignore[override]
        captured["config"] = config
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli.main,
            [
                "--url",
                "https://example.com",
                "--output",
                "out",
                "--user-agent",
                "Custom-UA",
                "--proxy-server",
                "http://user:pass@proxy.local:8080",
                "--proxy-bypass",
                "intranet.local",
                "--storage-state",
                "state.json",
                "--user-interaction-timeout",
                "5",
                "--headed",
            ],
        )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.user_agent == "Custom-UA"
    assert config.storage_state_path is not None
    assert config.storage_state_path.name == "state.json"
    assert config.user_interaction_timeout == 5
    assert config.headless is False
    assert config.proxy is not None
    proxy_dict = config.proxy.to_playwright_options()
    assert proxy_dict["server"].startswith("http://proxy.local")
    assert "bypass" in proxy_dict and "localhost" in proxy_dict["bypass"]


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
