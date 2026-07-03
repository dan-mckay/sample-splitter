import pytest
from typer.testing import CliRunner

from sample_splitter.cli import app

runner = CliRunner()


def test_help_lists_all_subcommands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "split" in result.stdout
    assert "name" in result.stdout


@pytest.mark.parametrize("command", ["scan", "split", "name"])
def test_stub_prints_resolved_settings(command, tmp_path):
    result = runner.invoke(app, [command, str(tmp_path)])

    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout
    assert "threshold_db" in result.stdout
    assert "taxonomy" in result.stdout
