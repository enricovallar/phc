import gdsfactory as gf
from phc_layout import LAYER_PHC, utils


def test_deduplicate_instances():
    basis = gf.Component("test_basis")
    circle = gf.components.circle(radius=0.1)
    # Add two circles at the exact same location
    basis << circle
    basis << circle
    assert len(basis.insts) == 2

    cleaned = utils.deduplicate_instances(basis)
    assert len(cleaned.insts) == 1


def test_get_component_hull():
    c = gf.Component("test_rect")
    c << gf.components.rectangle(size=(10, 5), layer=LAYER_PHC.ETCH)
    hull = utils.get_component_hull(c, layer=LAYER_PHC.ETCH, margin=1.0)
    assert isinstance(hull, gf.Component)
