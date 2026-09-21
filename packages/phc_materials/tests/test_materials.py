from pathlib import Path

import pytest
from phc_materials import (
    AnisotropicMaterial,
    IsotropicMaterial,
    MaterialRegistry,
    get_material,
    to_mpb_medium,
)


def test_isotropic_material():
    si = get_material("si")
    assert isinstance(si, IsotropicMaterial)
    assert si.index == 3.48
    assert pytest.approx(si.epsilon) == 3.48**2
    assert si.epsilon_diag == (si.epsilon, si.epsilon, si.epsilon)


def test_anisotropic_material():
    ln = get_material("lnoi_xcut")
    assert isinstance(ln, AnisotropicMaterial)
    assert ln.indices == (2.203, 2.286, 2.286)
    eps_x, eps_y, eps_z = ln.epsilon_diag
    assert pytest.approx(eps_x) == 2.203**2
    assert pytest.approx(eps_y) == 2.286**2
    assert pytest.approx(eps_z) == 2.286**2


def test_to_mpb_medium():
    si_med = to_mpb_medium("si")
    assert si_med is not None

    ln_med = to_mpb_medium("lnoi_xcut")
    assert ln_med is not None


def test_load_from_yaml():
    yaml_path = Path(__file__).parents[3] / "materials" / "materials.yaml"
    if yaml_path.is_file():
        reg = MaterialRegistry.from_yaml(yaml_path)
        assert "si" in reg
        assert "lnoi_xcut" in reg
        assert reg.get("si").index == 3.48
