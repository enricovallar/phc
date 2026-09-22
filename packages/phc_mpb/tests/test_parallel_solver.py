"""Unit tests for the concurrent multi-worker MPB band solver and anisotropic resolution."""

import meep as mp
import numpy as np
import pytest
from phc_materials import to_mpb_medium
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    get_high_symmetry_kpath,
    run_band_solver,
    run_parallel_band_solver,
)


def test_anisotropic_resolution_initialization():
    """Verifies that create_mode_solver constructs anisotropic resolution Vector3 correctly."""
    lat = create_lattice("square", dimension="3D_slab", supercell_z=4.0)
    k_pts = [mp.Vector3(0, 0, 0)]

    # 1. Using separate resolution and resolution_z
    ms1 = create_mode_solver(
        geometry_lattice=lat,
        geometry=[],
        k_points=k_pts,
        default_material="air",
        resolution=24,
        resolution_z=8,
        num_bands=2,
    )
    assert list(ms1.resolution) == [24.0, 24.0, 8.0]

    # 2. Using explicit 3-tuple
    ms2 = create_mode_solver(
        geometry_lattice=lat,
        geometry=[],
        k_points=k_pts,
        default_material="air",
        resolution=(20, 20, 6),
        num_bands=2,
    )
    assert list(ms2.resolution) == [20.0, 20.0, 6.0]

    # 3. Using mp.Vector3
    ms3 = create_mode_solver(
        geometry_lattice=lat,
        geometry=[],
        k_points=k_pts,
        default_material="air",
        resolution=mp.Vector3(18, 18, 5),
        num_bands=2,
    )
    assert list(ms3.resolution) == [18.0, 18.0, 5.0]


def test_parallel_solver_serial_equivalence():
    """Verifies that multi-worker execution yields identical frequencies to single-worker execution."""
    lat = create_lattice("hexagonal", dimension="2D")
    geom = [mp.Cylinder(radius=0.25, height=mp.inf, material=to_mpb_medium("air"))]
    k_pts, _labels, _indices = get_high_symmetry_kpath("hexagonal", k_density=2)

    # Serial solve (1 worker)
    res_serial = run_parallel_band_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=16,
        num_bands=2,
        polarization="te",
        dimension="2D",
        num_workers=1,
    )

    # Parallel solve (2 workers)
    res_parallel = run_parallel_band_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=16,
        num_bands=2,
        polarization="te",
        dimension="2D",
        num_workers=2,
    )

    freqs_serial = res_serial["freqs"]["te"]
    freqs_parallel = res_parallel["freqs"]["te"]

    assert freqs_serial.shape == freqs_parallel.shape
    assert np.allclose(freqs_serial, freqs_parallel, atol=1e-5)
    assert res_parallel["num_workers"] == 2


def test_parallel_solver_3d_slab_te_like():
    """Validates 3D slab parallel solve with even parity (zeven / te_like) and light line."""
    lat = create_lattice("hexagonal", dimension="3D_slab", supercell_z=3.0)
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, 0.5),
            material=to_mpb_medium("si"),
        ),
        mp.Cylinder(
            center=mp.Vector3(0, 0, 0),
            radius=0.25,
            height=0.5,
            material=to_mpb_medium("air"),
        ),
    ]
    k_pts, _labels, _indices = get_high_symmetry_kpath("hexagonal", k_density=1)

    res = run_parallel_band_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="air",
        resolution=12,
        resolution_z=4,
        num_bands=3,
        polarization="te_like",
        dimension="3D_slab",
        num_workers=2,
        cladding_index=1.0,
    )

    assert "te_like" in res["freqs"]
    freqs = res["freqs"]["te_like"]
    assert freqs.shape == (len(k_pts), 3)

    # Light line check
    assert "light_line" in res
    assert len(res["light_line"]) == len(k_pts)
    assert all(ll >= 0.0 for ll in res["light_line"])
    # At Gamma (k=0), light line must be 0. Γ is the second vertex (index = k_density + 1).
    gamma_idx = _indices[1]  # Γ is the second high-symmetry point
    assert np.isclose(res["light_line"][gamma_idx], 0.0, atol=1e-6)


def test_run_band_solver_delegation_to_parallel():
    """Verifies that run_band_solver automatically delegates when num_workers > 1."""
    lat = create_lattice("hexagonal", dimension="2D")
    geom = [mp.Cylinder(radius=0.2, height=mp.inf, material=to_mpb_medium("air"))]
    k_pts = [mp.Vector3(0, 0, 0), mp.Vector3(0, 0.5, 0)]

    ms = create_mode_solver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=k_pts,
        default_material="si",
        resolution=16,
        num_bands=2,
    )

    res = run_band_solver(
        ms=ms,
        polarization="te",
        dimension="2D",
        num_workers=2,
    )

    assert "te" in res["freqs"]
    assert res["freqs"]["te"].shape == (2, 2)
    assert hasattr(ms, "all_freqs")
    assert ms.all_freqs.shape == (2, 2)


def test_parallel_solver_validation_errors():
    """Verifies descriptive exception handling for invalid inputs."""
    lat = create_lattice("hexagonal", dimension="2D")

    # Empty k-points
    with pytest.raises(ValueError, match="k_points must be a non-empty sequence"):
        run_parallel_band_solver(
            geometry_lattice=lat,
            geometry=[],
            k_points=[],
            default_material="air",
        )

    # Invalid dimension
    with pytest.raises(ValueError, match="Unsupported dimension"):
        run_parallel_band_solver(
            geometry_lattice=lat,
            geometry=[],
            k_points=[mp.Vector3(0, 0, 0)],
            default_material="air",
            dimension="4D",  # type: ignore
        )


def test_plot_epsilon_3d_dual_plane(tmp_path):
    """Verifies that plot_epsilon on a 3D array generates a unified figure with xy and xz subplots."""
    from phc_mpb import plot_epsilon

    eps_3d = np.ones((24, 24, 40))
    eps_3d[:, :, 15:25] = 12.0  # Slab

    out_fig = tmp_path / "test_dual_plane_eps.png"
    fig = plot_epsilon(eps_3d, save_path=out_fig)

    assert out_fig.is_file()
    assert fig is not None
    # Must contain at least 2 primary axes (ax_xy and ax_xz)
    assert len(fig.axes) >= 2
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Mid-Plane" in t for t in titles)
    assert any("Cross-Section" in t for t in titles)
