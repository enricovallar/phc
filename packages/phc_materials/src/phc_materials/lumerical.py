import os
from pathlib import Path
from typing import Any

from phc_materials.models import (
    AnisotropicMaterial,
    IsotropicMaterial,
    MaterialSpec,
)


def load_material_database(fdtd_session: Any, mdf_path: str | Path) -> bool:
    """Loads a version-controlled .mdf material database into Lumerical."""
    mdf_path = str(mdf_path)
    if not os.path.isfile(mdf_path):
        raise FileNotFoundError(
            f"Lumerical .mdf database file not found at: {mdf_path}"
        )

    if hasattr(fdtd_session, "importmaterialdb"):
        fdtd_session.importmaterialdb(mdf_path)
        return True
    return False


def export_material_database(fdtd_session: Any, mdf_path: str | Path) -> bool:
    """Exports the current session's material database to a repository .mdf file."""
    mdf_path = str(mdf_path)
    os.makedirs(os.path.dirname(os.path.abspath(mdf_path)), exist_ok=True)
    if hasattr(fdtd_session, "exportmaterialdb"):
        fdtd_session.exportmaterialdb(mdf_path)
        return True
    return False


def add_material_to_lumerical(fdtd_session: Any, material: MaterialSpec) -> None:
    """Programmatically adds an isotropic or anisotropic material into Lumerical."""
    name = material.lumerical_name or material.name or "custom_material"

    # Check if session exists and is live
    if not hasattr(fdtd_session, "materialexists"):
        return

    if not fdtd_session.materialexists(name):
        mat = fdtd_session.addmaterial("Dielectric")
        fdtd_session.setmaterial(mat, "name", name)

    if isinstance(material, IsotropicMaterial):
        fdtd_session.setmaterial(name, "Anisotropy", 0)
        fdtd_session.setmaterial(name, "Permittivity", material.epsilon)
    elif isinstance(material, AnisotropicMaterial):
        eps_x, eps_y, eps_z = material.epsilon_diag
        fdtd_session.setmaterial(name, "Anisotropy", 1)  # 1 = diagonal anisotropy
        fdtd_session.setmaterial(name, "Permittivity", [eps_x, eps_y, eps_z])
