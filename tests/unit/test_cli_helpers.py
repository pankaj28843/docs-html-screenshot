import pytest
from click.testing import CliRunner

from docs_html_screenshot import cli


@pytest.mark.unit
def test_output_path_appends_suffix(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    (input_root / "a" / "b").mkdir(parents=True)
    source = input_root / "a" / "b" / "index.html"
    source.write_text("<html></html>")

    target = cli.output_path_for(source, input_root, output_root)

    assert target == output_root / "a" / "b" / "index.html-screenshot.png"


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
def test_click_help_runs_without_execution():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "docs-html-screenshot" in result.output
