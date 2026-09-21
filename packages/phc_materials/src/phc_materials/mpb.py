import numpy as np

from phc_materials.models import (
    AnisotropicMaterial,
    IsotropicMaterial,
    MaterialSpec,
)


def to_mpb_medium(material: MaterialSpec | str):
    """Converts a MaterialSpec or material key into a meep.Medium object.

    Handles both isotropic (scalar index) and anisotropic (epsilon_diag
    tensor + rotation) materials.
    """
    if isinstance(material, str):
        from phc_materials.registry import get_material

        material = get_material(material)

    import meep as mp

    if isinstance(material, IsotropicMaterial):
        return mp.Medium(index=material.index)

    elif isinstance(material, AnisotropicMaterial):
        eps_x, eps_y, eps_z = material.epsilon_diag
        medium = mp.Medium(epsilon_diag=mp.Vector3(eps_x, eps_y, eps_z))
        if material.rotation_deg != 0.0:
            medium = medium.rotate(
                mp.Vector3(0, 0, 1), np.deg2rad(material.rotation_deg)
            )
        return medium

    raise TypeError(f"Unsupported material spec type: {type(material)}")
