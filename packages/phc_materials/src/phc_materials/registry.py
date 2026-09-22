from pathlib import Path
from typing import Any

import yaml

from phc_materials.models import (
    AnisotropicMaterial,
    IsotropicMaterial,
    MaterialSpec,
)

DEFAULT_MATERIALS: dict[str, dict[str, Any]] = {
    "air": {
        "type": "isotropic",
        "index": 1.0,
        "lumerical_name": "etch",
        "description": "Air / Vacuum",
    },
    "si": {
        "type": "isotropic",
        "index": 3.48,
        "lumerical_name": "Si (Silicon) - Palik",
        "description": "Silicon at 1550 nm",
    },
    "sio2": {
        "type": "isotropic",
        "index": 1.444,
        "lumerical_name": "SiO2 (Glass) - Palik",
        "description": "Silicon Dioxide at 1550 nm",
    },
    "sin": {
        "type": "isotropic",
        "index": 2.00,
        "lumerical_name": "Si3N4 - Philip",
        "description": "Silicon Nitride at 1550 nm",
    },
    "lnoi_xcut": {
        "type": "anisotropic",
        "indices": [2.203, 2.286, 2.286],
        "rotation_deg": 0.0,
        "lumerical_name": "LN_Xcut",
        "description": "X-cut Lithium Niobate (optic axis along X)",
    },
    "lnoi_zcut": {
        "type": "anisotropic",
        "indices": [2.286, 2.286, 2.203],
        "rotation_deg": 0.0,
        "lumerical_name": "LN_Zcut",
        "description": "Z-cut Lithium Niobate (optic axis along Z)",
    },
    "batio3": {
        "type": "anisotropic",
        "indices": [2.35, 2.35, 2.30],
        "rotation_deg": 0.0,
        "lumerical_name": "BaTiO3_Custom",
        "description": "Barium Titanate thin film",
    },
    "inp": {
        "type": "isotropic",
        "index": 3.1,
        "lumerical_name": "InP (Indium Phosphide) - Palik",
        "description": "Indium Phosphide at 1550 nm",
    },
}


class MaterialRegistry:
    """Registry storing optical material definitions."""

    def __init__(self, materials_dict: dict[str, Any] | None = None) -> None:
        self._materials: dict[str, MaterialSpec] = {}
        raw = materials_dict if materials_dict is not None else DEFAULT_MATERIALS
        for key, val in raw.items():
            self.register(key, val)

    def register(self, key: str, data: dict[str, Any] | MaterialSpec) -> None:
        if isinstance(data, (IsotropicMaterial, AnisotropicMaterial)):
            self._materials[key] = data
            return

        mat_type = data.get("type", "isotropic")
        item_data = dict(data)
        item_data.setdefault("name", key)

        if mat_type == "anisotropic":
            self._materials[key] = AnisotropicMaterial(**item_data)
        elif mat_type == "isotropic":
            self._materials[key] = IsotropicMaterial(**item_data)
        else:
            raise ValueError(f"Unknown material type '{mat_type}' for '{key}'")

    def get(self, key: str) -> MaterialSpec:
        if key not in self._materials:
            raise KeyError(
                f"Material '{key}' not found. Available: {list(self._materials.keys())}"
            )
        return self._materials[key]

    def __contains__(self, key: str) -> bool:
        return key in self._materials

    def __iter__(self):
        return iter(self._materials)

    def keys(self) -> list[str]:
        return list(self._materials.keys())

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MaterialRegistry":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        materials_section = data.get("materials", data)
        return cls(materials_section)


# Global default registry instance
default_registry = MaterialRegistry()


def get_material(key: str) -> MaterialSpec:
    return default_registry.get(key)
