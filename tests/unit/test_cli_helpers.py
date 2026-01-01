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
def test_cli_requires_input_option():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--output", "/tmp/out"])

    assert result.exit_code != 0
    assert "Missing option '--input'" in result.output


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
def test_start_stop_server(tmp_path):
    input_dir = tmp_path / "html"
    input_dir.mkdir()
    (input_dir / "index.html").write_text("<html><body>Test</body></html>")

    server, thread, port = cli._start_server(input_dir)

    assert thread.is_alive()
    assert 1024 <= port <= 65535

    cli._stop_server(server, thread)

    assert not thread.is_alive()
