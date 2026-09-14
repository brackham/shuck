from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from shuck.resources import REQUIRED_SPEXTOOL_ASSETS

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.packaging
def test_wheel_contains_resources_and_installed_locator_finds_them(tmp_path: Path) -> None:
    distribution_directory = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(distribution_directory),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    wheel = next(distribution_directory.glob("*.whl"))
    expected = {f"shuck/data/spextool/{relative}" for relative in REQUIRED_SPEXTOOL_ASSETS}
    expected.update(
        {
            "shuck/data/spextool/README.md",
            "shuck/data/spextool/SHA256SUMS",
        }
    )
    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert expected <= members
        assert any(member.endswith(".dist-info/licenses/LICENSE") for member in members)
        assert any(
            member.endswith(".dist-info/licenses/THIRD_PARTY_NOTICES.md") for member in members
        )

    target = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-compile",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
    )
    environment = os.environ.copy()
    environment.pop("SPEXTOOL5_DIR", None)
    environment["PYTHONPATH"] = str(target)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from shuck.resources import REQUIRED_SPEXTOOL_ASSETS, spextool_root; "
                "root = spextool_root(); "
                "assert all((root / item).is_file() for item in REQUIRED_SPEXTOOL_ASSETS)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
