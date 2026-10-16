"""Installed-package smoke tests; no source-path injection."""

from importlib.metadata import requires, version
from pathlib import Path
import subprocess
import sys
import tomllib

import mdac2


def test_installed_version_matches_source():
    assert version("mdac2") == mdac2.__version__ == "0.1.0"


def test_core_has_no_runtime_dependencies():
    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text("utf-8"))
    assert config["project"]["dependencies"] == []
    assert all("extra ==" in requirement for requirement in (requires("mdac2") or []))


def test_import_outside_checkout_is_lightweight(tmp_path):
    code = (
        "import mdac2, sys; "
        "assert mdac2.__version__ == '0.1.0'; "
        "assert not {'torch', 'fastapi', 'librosa', 'pytest'} & sys.modules.keys()"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
