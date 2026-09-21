from pathlib import Path

import pytest
from phc_utils.io import export_gds


class MockComponent:
    def __init__(self, name: str = "mock"):
        self.name = name

    def write_gds(self, filepath: str) -> None:
        Path(filepath).write_text("mock gds content")


def test_export_gds_success(tmp_path: Path):
    target = tmp_path / "sub" / "cell.gds"
    comp = MockComponent()
    res = export_gds(comp, target)
    assert res == target
    assert target.is_file()
    assert target.read_text() == "mock gds content"


def test_export_gds_overwrite(tmp_path: Path):
    target = tmp_path / "cell.gds"
    target.write_text("old content")
    comp = MockComponent()

    # Should raise FileExistsError if overwrite=False
    with pytest.raises(FileExistsError):
        export_gds(comp, target, overwrite=False)

    # Should succeed if overwrite=True
    export_gds(comp, target, overwrite=True)
    assert target.read_text() == "mock gds content"


def test_export_gds_invalid_type(tmp_path: Path):
    with pytest.raises(TypeError):
        export_gds("not_a_component", tmp_path / "cell.gds")
