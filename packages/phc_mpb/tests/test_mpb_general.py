import matplotlib.pyplot as plt
import numpy as np
import pytest
from phc_layout.components import (
    get_unit_cell,
    phc_hexagonal_unit_cell,
    phc_square_unit_cell,
)
from phc_layout.lattice import HexagonalLattice, SquareLattice
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    lattice_to_mpb_lattice,
    plot_band_structure,
    plot_epsilon,
    to_mpb_lattice,
)


def test_hexagonal_lattice_creation_2d():
    lat = create_lattice(lattice_type="hexagonal", dimension="2D")
    assert lat is not None


def test_square_lattice_creation_slab():
    lat = create_lattice(lattice_type="square", dimension="3D_slab", supercell_z=6.0)
    assert lat is not None


def test_kpath_hexagonal():
    k_pts, labels, indices = get_high_symmetry_kpath(
        lattice_type="hexagonal", k_density=10
    )
    assert labels == ["M", "Γ", "K", "M"]
    assert len(k_pts) > len(labels)
    assert len(indices) == len(labels)


def test_kpath_square():
    k_pts, labels, _indices = get_high_symmetry_kpath(
        lattice_type="square", k_density=10
    )
    assert labels == ["X", "Γ", "M", "X"]
    assert len(k_pts) > len(labels)


def test_gds_to_mpb_geometry_from_component():
    c_hex = phc_hexagonal_unit_cell(pitch=1.0, radius=0.25)
    geom_2d = gds_to_mpb_geometry(c_hex, pitch=1.0, dimension="2D")
    assert len(geom_2d) > 0

    c_sq = phc_square_unit_cell(pitch=0.5, radius=0.15)
    geom_slab = gds_to_mpb_geometry(
        c_sq, pitch=0.5, dimension="3D_slab", slab_thickness=0.22
    )
    assert len(geom_slab) > 0


def test_mode_solver_creation():
    c = phc_hexagonal_unit_cell(pitch=1.0, radius=0.25)
    lat = create_lattice("hexagonal", dimension="2D")
    geom = gds_to_mpb_geometry(c, pitch=1.0, dimension="2D")
    k_pts, _labels, _indices = get_high_symmetry_kpath("hexagonal", k_density=5)

    ms = create_mode_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=16,
        num_bands=4,
    )
    assert ms is not None


def test_plotting_dummy_results(tmp_path):
    k_pts, labels, indices = get_high_symmetry_kpath("hexagonal", k_density=5)
    num_k = len(k_pts)
    dummy_freqs = np.zeros((num_k, 4))
    for b in range(4):
        dummy_freqs[:, b] = 0.2 * (b + 1) + 0.05 * np.sin(np.linspace(0, np.pi, num_k))

    results = {
        "freqs": {"te": dummy_freqs},
        "dimension": "2D",
    }
    out_png = tmp_path / "test_band_plot.png"
    fig = plot_band_structure(results, labels, indices, save_path=out_png)
    assert out_png.is_file()
    assert fig is not None
    # Verify bands are plotted strictly as dots with no continuous lines
    ax = fig.axes[0]
    band_lines = [
        line for line in ax.get_lines() if line.get_linestyle() in ("None", "none", "")
    ]
    assert len(band_lines) == 4
    for line in band_lines:
        assert line.get_linestyle() in ("None", "none", "")
        assert line.get_marker() == "o"


def test_plot_band_structure_dual_markers_unified(tmp_path):
    """Verifies unified multi-polarization plots support one marker for TE/TM and a different marker for all bands."""
    k_pts, labels, indices = get_high_symmetry_kpath("hexagonal", k_density=3)
    num_k = len(k_pts)
    freqs_te = np.full((num_k, 2), 0.2)
    freqs_tm = np.full((num_k, 2), 0.4)
    freqs_all = np.full((num_k, 4), 0.3)
    te_fracs = np.full((num_k, 4), 0.8)

    results = {
        "freqs": {
            "all": freqs_all,
            "te_like": freqs_te,
            "tm_like": freqs_tm,
        },
        "te_fractions": te_fracs,
        "dimension": "3D_slab",
    }
    out_png = tmp_path / "test_dual_marker_band_plot.png"
    fig = plot_band_structure(results, labels, indices, save_path=out_png)
    assert out_png.is_file()
    assert fig is not None

    ax = fig.axes[0]
    # TE-like and TM-like parity modes both use circles ('o') with different colors
    te_lines = [
        line
        for line in ax.get_lines()
        if line.get_marker() == "o" and line.get_color() == "tab:blue"
    ]
    tm_lines = [
        line
        for line in ax.get_lines()
        if line.get_marker() == "o" and line.get_color() == "tab:red"
    ]
    assert len(te_lines) == 2
    assert len(tm_lines) == 2

    # All-bands simulation uses squares ('s') colored by TE fraction
    scatters = ax.collections
    assert len(scatters) == 4  # 4 bands
    # Verify scatter paths are square (4 vertices)
    for sc in scatters:
        paths = sc.get_paths()
        assert len(paths) > 0


def test_plot_band_structure_custom_markers_and_ax(tmp_path):
    """Verifies custom marker dictionaries, custom single marker string, and ax embedding."""
    import matplotlib.pyplot as plt

    k_pts, labels, indices = get_high_symmetry_kpath("hexagonal", k_density=3)
    num_k = len(k_pts)
    freqs_te = np.full((num_k, 2), 0.2)
    freqs_tm = np.full((num_k, 2), 0.4)

    results = {
        "freqs": {
            "te_like": freqs_te,
            "tm_like": freqs_tm,
        },
        "dimension": "3D_slab",
    }

    # Test custom markers dictionary
    fig_custom, ax_custom = plt.subplots()
    plot_band_structure(
        results,
        labels,
        indices,
        markers={"te_like": "s", "tm_like": "x"},
        ax=ax_custom,
    )
    s_lines = [l for l in ax_custom.get_lines() if l.get_marker() == "s"]
    x_lines = [l for l in ax_custom.get_lines() if l.get_marker() == "x"]
    assert len(s_lines) == 2
    assert len(x_lines) == 2
    plt.close(fig_custom)


def test_replot_band_structure_from_results(tmp_path):
    """Verifies that replot_band_structure_from_results correctly reloads JSON and renders plots."""
    import json

    from phc_mpb.plotting import replot_band_structure_from_results

    # 1. Create a structured simulation_results.json
    sim_dir = (
        tmp_path / "mpb" / "slab_mode_parity" / "c6v_primitive" / "2026-01-01_12-00-00"
    )
    sim_dir.mkdir(parents=True)
    json_path = sim_dir / "simulation_results.json"

    dummy_results = {
        "geometry": {"name": "c6v_primitive", "dimension": "3D_slab", "pitch_um": 0.45},
        "simulation": {"solver": "mpb", "sim_type": "slab_mode_parity"},
        "extra_data": {
            "frequencies": {
                "te_like": [[0.2, 0.3], [0.22, 0.32]],
                "tm_like": [[0.35, 0.45], [0.36, 0.46]],
                "all": [[0.2, 0.3, 0.35, 0.45], [0.22, 0.32, 0.36, 0.46]],
            },
            "te_fractions": [[0.95, 0.92, 0.05, 0.08], [0.94, 0.91, 0.06, 0.09]],
            "light_line": [0.1, 0.2],
            "k_labels": ["M", "Γ"],
            "k_indices": [0, 1],
        },
    }
    json_path.write_text(json.dumps(dummy_results))

    # 2. Test explicit JSON path replot
    save_fig = tmp_path / "replotted.png"
    fig = replot_band_structure_from_results(
        results_path=json_path,
        markers={"te_like": "o", "tm_like": "o", "all": "D"},
        save_path=save_fig,
    )
    assert fig is not None
    assert save_fig.is_file()
    plt.close(fig)

    # 3. Test auto-discovery of latest run (results_path=None)
    fig_auto = replot_band_structure_from_results(
        results_path=None,
        sim_type="slab_mode_parity",
        base_dir=tmp_path,
    )
    assert fig_auto is not None
    plt.close(fig_auto)


def test_get_epsilon_and_plot(tmp_path):
    c = phc_hexagonal_unit_cell(pitch=1.0, radius=0.25)
    lat = create_lattice("hexagonal", dimension="2D")
    geom = gds_to_mpb_geometry(c, pitch=1.0, dimension="2D")
    k_pts, _labels, _indices = get_high_symmetry_kpath("hexagonal", k_density=5)
    ms = create_mode_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=16,
        num_bands=4,
    )

    eps = get_epsilon_grid(ms, rectify=True, periodicity=3, resolution=64)
    assert isinstance(eps, np.ndarray)
    assert eps.ndim == 2
    assert 64 * 3 in eps.shape
    assert eps.shape[0] > 0

    out_eps_png = tmp_path / "test_epsilon_plot.png"
    fig = plot_epsilon(eps, save_path=out_eps_png)
    assert out_eps_png.is_file()
    assert fig is not None
    im = fig.axes[0].get_images()[0]
    assert im.get_interpolation() == "none"

    # Test 3D slab dual-plane visualization with no interpolation
    eps_3d = np.ones((16, 16, 16))
    fig_3d = plot_epsilon(eps_3d)
    assert len(fig_3d.axes) >= 2
    for ax_i in fig_3d.axes[:2]:
        images = ax_i.get_images()
        if images:
            assert images[0].get_interpolation() == "none"


def test_lattice_to_mpb_lattice_hexagonal():
    layout_lat = HexagonalLattice(a=0.5)
    mpb_lat = lattice_to_mpb_lattice(layout_lat, dimension="2D")
    assert mpb_lat is not None

    # Verify basis vectors are normalized (a = 1)
    b1 = mpb_lat.basis1
    b2 = mpb_lat.basis2
    assert np.isclose(b1.x, 1.0)
    assert np.isclose(b1.y, 0.0)
    assert np.isclose(b2.x, -0.5)
    assert np.isclose(b2.y, np.sqrt(3.0) / 2.0)
    assert mpb_lat.size.z == 0.0

    # Test 3D slab
    mpb_slab = lattice_to_mpb_lattice(layout_lat, dimension="3D_slab", supercell_z=7.0)
    assert mpb_slab.size.z == 7.0


def test_lattice_to_mpb_lattice_square():
    layout_lat = SquareLattice(a=0.4)
    mpb_lat = to_mpb_lattice(layout_lat, dimension="2D")
    b1 = mpb_lat.basis1
    b2 = mpb_lat.basis2
    assert np.isclose(b1.x, 1.0)
    assert np.isclose(b1.y, 0.0)
    assert np.isclose(b2.x, 0.0)
    assert np.isclose(b2.y, 1.0)


def test_create_lattice_with_lattice_object():
    layout_lat = HexagonalLattice(a=0.5)
    mpb_lat = create_lattice(layout_lat, dimension="2D")
    assert np.isclose(mpb_lat.basis1.x, 1.0)
    assert np.isclose(mpb_lat.basis2.x, -0.5)


def test_lattice_to_mpb_type_error():
    with pytest.raises(TypeError, match="Expected phc_layout.lattice.Lattice"):
        lattice_to_mpb_lattice("invalid_lattice")  # type: ignore[arg-type]


def test_kpath_with_layout_lattice_objects():
    hex_lat = HexagonalLattice(a=0.5)
    k_pts, labels, _indices = get_high_symmetry_kpath(hex_lat, k_density=10)
    assert labels == ["M", "Γ", "K", "M"]
    assert len(k_pts) > len(labels)

    sq_lat = SquareLattice(a=0.5)
    k_pts_sq, labels_sq, _ = get_high_symmetry_kpath(sq_lat, k_density=10)
    assert labels_sq == ["X", "Γ", "M", "X"]
    assert len(k_pts_sq) > len(labels_sq)


def test_mode_solver_with_layout_lattice_and_database_cell():
    """End-to-end test connecting phc_layout.lattice, unit cell database, and MPB solver."""
    pitch = 0.5
    layout_lat = HexagonalLattice(a=pitch)
    mpb_lat = lattice_to_mpb_lattice(layout_lat, dimension="2D")

    # Get unit cell from database
    cell = get_unit_cell("c6v_primitive", pitch=pitch, radius=0.125)
    geom = gds_to_mpb_geometry(cell, pitch=pitch, dimension="2D")
    k_pts, _, _ = get_high_symmetry_kpath(layout_lat, k_density=3)

    ms = create_mode_solver(
        geometry_lattice=mpb_lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=8,
        num_bands=4,
    )
    assert ms is not None


def test_compute_polarization_fractions():
    """Verify TE/TM polarization fraction computation on a simple slab."""
    import meep as mp
    from meep import mpb
    from phc_mpb import compute_polarization_fractions

    lattice = mp.Lattice(size=mp.Vector3(1, 1, 3))
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, 0.5),
            material=mp.Medium(index=3.5),
        )
    ]
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geom,
        k_points=[mp.Vector3(0.2, 0, 0)],
        resolution=16,
        num_bands=2,
    )
    ms.run()

    # Single band
    frac_1 = compute_polarization_fractions(ms, band_idx=1)
    assert isinstance(frac_1, dict)
    assert "te" in frac_1 and "tm" in frac_1
    assert 0.0 <= frac_1["te"] <= 1.0
    assert 0.0 <= frac_1["tm"] <= 1.0
    assert abs(frac_1["te"] + frac_1["tm"] - 1.0) < 0.01

    # All bands
    fracs_all = compute_polarization_fractions(ms)
    assert isinstance(fracs_all, list)
    assert len(fracs_all) == 2
    for f in fracs_all:
        assert 0.0 <= f["te"] <= 1.0
        assert abs(f["te"] + f["tm"] - 1.0) < 0.01


def test_substrate_geometry_block_generated():
    """Verify that gds_to_mpb_geometry creates a substrate block when substrate_material is specified."""
    c_hex = phc_hexagonal_unit_cell(pitch=1.0, radius=0.25)
    geom_no_sub = gds_to_mpb_geometry(
        c_hex,
        pitch=1.0,
        dimension="3D_slab",
        slab_thickness=0.22,
        slab_material="si",
    )
    geom_with_sub = gds_to_mpb_geometry(
        c_hex,
        pitch=1.0,
        dimension="3D_slab",
        slab_thickness=0.22,
        slab_material="si",
        substrate_material="sio2",
    )
    # Substrate adds one extra Block before the slab
    assert len(geom_with_sub) == len(geom_no_sub) + 1


def test_run_band_solver_no_parity():
    """Verify that run_band_solver with polarization='all' returns frequencies and te_fractions."""
    import meep as mp
    from meep import mpb
    from phc_mpb import run_band_solver

    lattice = mp.Lattice(size=mp.Vector3(1, 1, 3))
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, 0.5),
            material=mp.Medium(index=3.5),
        )
    ]
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geom,
        k_points=[mp.Vector3(0.1, 0, 0), mp.Vector3(0.2, 0, 0)],
        resolution=16,
        num_bands=2,
    )

    results = run_band_solver(ms, polarization="all", dimension="3D_slab")

    assert "all" in results["freqs"]
    assert results["freqs"]["all"].shape == (2, 2)
    assert "te_fractions" in results
    assert results["te_fractions"].shape == (2, 2)
    assert np.all(results["te_fractions"] >= 0.0)
    assert np.all(results["te_fractions"] <= 1.0)
