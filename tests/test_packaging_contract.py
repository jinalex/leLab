from __future__ import annotations

from pathlib import Path
from tomllib import loads

ROOT = Path(__file__).parents[1]
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"


def test_python_and_base_dependencies_are_reproducibly_pinned() -> None:
    project = loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    assert "pygame==2.6.1" in dependencies
    assert any(
        dependency.startswith("lerobot[core_scripts,feetech,training]")
        and dependency.endswith(f"@{LEROBOT_COMMIT}")
        for dependency in dependencies
    )
    assert not any("hidapi" in dependency.casefold() for dependency in dependencies)
