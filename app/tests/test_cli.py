import pytest

from capp.cli import main


def test_cli_help_builds_without_conflicting_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "model-calibrate" in output
    assert "machine-map" in output


def test_machine_map_help_builds_without_conflicting_options(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["machine-map", "--help"])

    assert exc.value.code == 0
    assert "--voxel-spacing" in capsys.readouterr().out
