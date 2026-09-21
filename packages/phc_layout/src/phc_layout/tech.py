import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack


class LayerMapPhC(gf.technology.LayerMap):
    ETCH: tuple[int, int] = (1, 0)
    SLAB: tuple[int, int] = (2, 0)


LAYER_PHC = LayerMapPhC


def get_layer_stack(
    slab_thickness: float = 0.22,
    slab_material: str = "si",
    etch_material: str = "air",
) -> LayerStack:
    """Returns the default LayerStack for Photonic Crystal structures."""
    return LayerStack(
        layers={
            "slab": LayerLevel(
                layer=LAYER_PHC.SLAB,
                thickness=slab_thickness,
                zmin=0.0,
                material=slab_material,
                mesh_order=2,
            ),
            "etch": LayerLevel(
                layer=LAYER_PHC.ETCH,
                thickness=slab_thickness,
                zmin=0.0,
                material=etch_material,
                mesh_order=1,
            ),
        }
    )


__all__ = ["LAYER_PHC", "LayerMapPhC", "get_layer_stack"]
