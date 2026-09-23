import gdsfactory as gf
import phc_layout.components as cmp


def test_hex_phc():
    n_rows = 15
    n_cols = 20
    c = cmp.phc.lattice_patch_phc_C6v_rect(n_rows=n_rows, n_cols=n_cols)

    # Verify it is a valid Component
    assert isinstance(c, gf.Component)
    assert len(c.insts) == n_rows * n_cols


def test_triangular_basis():
    basis = cmp.basis.triangular_basis()
    assert isinstance(basis, gf.Component)
    # 1 center circle + 6 edge circles = 7 instances
    assert len(basis.insts) == 7


def test_phc_get_component_hull():
    c = gf.Component("test_hull_rect")
    c << gf.components.rectangle(size=(10, 5))
    hull = cmp.phc.get_component_hull(c, margin=1.0)
    assert isinstance(hull, gf.Component)
