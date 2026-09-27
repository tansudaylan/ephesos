from pathlib import Path

from ephesos.cli import run_output_example


def test_run_output_example_uses_default_output(monkeypatch):
    """The shared example CLI preserves the script-specific default path."""

    output_path = Path("default.gif")
    received_paths = []
    monkeypatch.setattr("sys.argv", ["example.py"])

    exit_code = run_output_example(received_paths.append, output_path, "Example")

    assert exit_code == 0
    assert received_paths == [output_path]


def test_run_output_example_uses_explicit_output(monkeypatch, tmp_path):
    """The shared example CLI forwards a user-selected output path."""

    output_path = tmp_path / "animation.gif"
    received_paths = []
    monkeypatch.setattr("sys.argv", ["example.py", "--output", str(output_path)])

    exit_code = run_output_example(received_paths.append, Path("default.gif"), "Example")

    assert exit_code == 0
    assert received_paths == [output_path]