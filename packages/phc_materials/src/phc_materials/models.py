from typing import Annotated, Literal

from pydantic import BaseModel, Field


class BaseMaterial(BaseModel):
    name: str | None = None
    description: str | None = None
    lumerical_name: str | None = None


class IsotropicMaterial(BaseMaterial):
    type: Literal["isotropic"] = "isotropic"
    index: float

    @property
    def epsilon(self) -> float:
        return self.index**2

    @property
    def epsilon_diag(self) -> tuple[float, float, float]:
        eps = self.epsilon
        return (eps, eps, eps)


class AnisotropicMaterial(BaseMaterial):
    type: Literal["anisotropic"] = "anisotropic"
    indices: tuple[float, float, float] = Field(
        ..., description="Principal refractive indices [nx, ny, nz]"
    )
    rotation_deg: float = Field(
        default=0.0, description="In-plane rotation angle in degrees"
    )

    @property
    def epsilon_diag(self) -> tuple[float, float, float]:
        nx, ny, nz = self.indices
        return (nx**2, ny**2, nz**2)


MaterialSpec = Annotated[
    IsotropicMaterial | AnisotropicMaterial,
    Field(discriminator="type"),
]
