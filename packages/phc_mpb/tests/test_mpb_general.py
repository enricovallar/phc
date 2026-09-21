import numpy as np
from phc_layout.components import (
    phc_hexagonal_unit_cell,
    phc_square_unit_cell,
)
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    plot_band_structure,
    plot_epsilon,
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
    assert labels == ["Γ", "M", "K", "Γ"]
    assert len(k_pts) > len(labels)
    assert len(indices) == len(labels)


def test_kpath_square():
    k_pts, labels, _indices = get_high_symmetry_kpath(
        lattice_type="square", k_density=10
    )
    assert labels == ["Γ", "X", "M", "Γ"]
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
