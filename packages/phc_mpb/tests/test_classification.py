"""Unit tests for modal physics classification in phc_mpb.classification."""

import numpy as np
import pytest
from phc_mpb.classification import (
    ModeClassification,
    classify_modes,
    summarize_mode_physics,
)


def test_mode_classification_enum():
    """Verifies ModeClassification enum values and string representations."""
    assert ModeClassification.GUIDED.value == "guided"
    assert ModeClassification.LEAKY.value == "leaky"
    assert ModeClassification.RADIATION.value == "radiation"
    assert str(ModeClassification.GUIDED) == "guided"


def test_classify_modes():
    """Verifies mode classification based on light line and confinement thresholds."""
    # 3 k-points, 3 bands
    # Light line frequencies: [0.3, 0.4, 0.5]
    light_line = [0.3, 0.4, 0.5]
    freqs = np.array(
        [
            [0.2, 0.25, 0.45],  # k0: band 1 & 2 below light line (0.3), band 3 above
            [0.35, 0.42, 0.60],  # k1: band 1 below light line (0.4), band 2 & 3 above
            [0.48, 0.55, 0.70],  # k2: band 1 below light line (0.5), band 2 & 3 above
        ]
    )
    confinements = np.array(
        [
            [
                0.90,
                0.15,
                0.75,
            ],  # k0: b1 guided, b2 below light line but low conf (<0.2) -> radiation, b3 above + high conf -> leaky
            [
                0.85,
                0.60,
                0.08,
            ],  # k1: b1 guided, b2 leaky, b3 above + low conf (<0.2) -> radiation
            [0.92, 0.10, 0.70],  # k2: b1 guided, b2 radiation, b3 leaky
        ]
    )

    classes = classify_modes(freqs, confinements, light_line, cutoff=0.20)
    assert classes.shape == (3, 3)

    # k0
    assert classes[0, 0] == ModeClassification.GUIDED.value
    assert classes[0, 1] == ModeClassification.RADIATION.value
    assert classes[0, 2] == ModeClassification.LEAKY.value

    # k1
    assert classes[1, 0] == ModeClassification.GUIDED.value
    assert classes[1, 1] == ModeClassification.LEAKY.value
    assert classes[1, 2] == ModeClassification.RADIATION.value

    # k2
    assert classes[2, 0] == ModeClassification.GUIDED.value
    assert classes[2, 1] == ModeClassification.RADIATION.value
    assert classes[2, 2] == ModeClassification.LEAKY.value


def test_classify_modes_shape_mismatch():
    """Verifies that mismatched array shapes raise ValueError."""
    freqs = np.ones((5, 4))
    confinements = np.ones((5, 3))
    light_line = [0.5] * 5

    with pytest.raises(ValueError, match="Shape mismatch"):
        classify_modes(freqs, confinements, light_line)

    with pytest.raises(ValueError, match="does not match"):
        classify_modes(freqs, np.ones((5, 4)), [0.5] * 4)


def test_summarize_mode_physics():
    """Verifies statistical summary of mode physics."""
    light_line = [0.3, 0.4]
    freqs = np.array(
        [
            [0.2, 0.4],
            [0.35, 0.5],
        ]
    )
    confinements = np.array(
        [
            [0.90, 0.70],  # k0: b1 guided, b2 leaky
            [0.80, 0.05],  # k1: b1 guided, b2 radiation
        ]
    )

    summary = summarize_mode_physics(freqs, confinements, light_line, cutoff=0.20)
    assert summary["total_modes"] == 4
    assert summary["num_guided"] == 2
    assert summary["num_leaky"] == 1
    assert summary["num_radiation"] == 1
    assert pytest.approx(summary["guided_ratio"], 0.01) == 0.50
    assert pytest.approx(summary["leaky_ratio"], 0.01) == 0.25
    assert pytest.approx(summary["radiation_ratio"], 0.01) == 0.25
    assert pytest.approx(summary["mean_confinement_guided"], 0.01) == 0.85
    assert pytest.approx(summary["mean_confinement_leaky"], 0.01) == 0.70
    assert len(summary["classification_matrix"]) == 2


def test_compute_polarization_fractions_with_slab_core_masking():
    """Verifies that slab_thickness and z_center core masking accurately computes polarization fractions."""
    import meep as mp
    from meep import mpb
    from phc_mpb.classification import (
        compute_modal_metrics,
        compute_polarization_fractions,
        compute_slab_confinement,
    )

    # 3D supercell with Si slab (center z=0, thickness=0.5, eps=12.25)
    # and SiO2 substrate (center z=-1.25, thickness=1.5, eps=2.07)
    lat = mp.Lattice(size=mp.Vector3(1, 1, 4))
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0, -1.25),
            size=mp.Vector3(mp.inf, mp.inf, 1.5),
            material=mp.Medium(index=1.44),
        ),
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, 0.5),
            material=mp.Medium(index=3.5),
        ),
    ]
    ms = mpb.ModeSolver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=[mp.Vector3(0.2, 0, 0)],
        resolution=16,
        num_bands=2,
    )
    ms.run()

    # 1. With core masking: slab_thickness=0.5, z_center=0.0
    frac_core = compute_polarization_fractions(
        ms, band_idx=1, slab_thickness=0.5, z_center=0.0
    )
    assert isinstance(frac_core, dict)
    assert "te" in frac_core and "tm" in frac_core
    assert 0.0 <= frac_core["te"] <= 1.0
    assert abs(frac_core["te"] + frac_core["tm"] - 1.0) < 1e-4

    # 2. Confinement with core masking
    conf_core = compute_slab_confinement(
        ms, band_idx=1, slab_thickness=0.5, z_center=0.0
    )
    assert isinstance(conf_core, float)
    assert 0.0 <= conf_core <= 1.0

    # 3. All bands metrics with core masking
    metrics_all = compute_modal_metrics(ms, slab_thickness=0.5, z_center=0.0)
    assert isinstance(metrics_all, list)
    assert len(metrics_all) == 2
    for m in metrics_all:
        assert "te" in m and "tm" in m and "confinement" in m
        assert 0.0 <= m["te"] <= 1.0
        assert 0.0 <= m["tm"] <= 1.0
        assert 0.0 <= m["confinement"] <= 1.0
        assert abs(m["te"] + m["tm"] - 1.0) < 1e-4

    # 4. Backward compatibility: without slab_thickness (None)
    frac_compat = compute_polarization_fractions(ms, band_idx=1)
    assert isinstance(frac_compat, dict)
    assert 0.0 <= frac_compat["te"] <= 1.0


def test_mode_overlap_and_tracking():
    """Verifies mode overlap integral, overlap matrix, and mode tracking."""
    import meep as mp
    from meep import mpb
    from phc_mpb.classification import (
        compute_mode_overlap,
        compute_mode_overlap_matrix,
        track_modes_by_overlap,
    )

    lat = mp.Lattice(size=mp.Vector3(1, 1, 3))
    geom = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(mp.inf, mp.inf, 0.4),
            material=mp.Medium(index=2.5),
        )
    ]
    ms = mpb.ModeSolver(
        geometry_lattice=lat,
        geometry=geom,
        k_points=[mp.Vector3(0.25, 0, 0)],
        resolution=12,
        num_bands=3,
    )
    ms.run()

    # 1. Self-overlap must be 1.0
    for field_type in ("electric_displacement", "electric", "magnetic"):
        ov_self = compute_mode_overlap(ms, 1, ms, 1, field=field_type)
        assert pytest.approx(ov_self, abs=1e-4) == 1.0

    # 2. Orthogonality of distinct bands
    ov_12 = compute_mode_overlap(ms, 1, ms, 2, field="electric_displacement")
    assert ov_12 < 0.05

    # 3. Overlap with slab masking
    ov_slab = compute_mode_overlap(
        ms, 1, ms, 1, field="electric_displacement", slab_thickness=0.4
    )
    assert pytest.approx(ov_slab, abs=1e-3) == 1.0

    # 4. Overlap matrix computation
    mat = compute_mode_overlap_matrix(ms, ms, bands_ref=[1, 2], bands_target=[1, 2, 3])
    assert mat.shape == (2, 3)
    assert pytest.approx(mat[0, 0], abs=1e-4) == 1.0
    assert pytest.approx(mat[1, 1], abs=1e-4) == 1.0
    assert mat[0, 1] < 0.05
    assert mat[1, 0] < 0.05

    # 5. Mode tracking
    tracking = track_modes_by_overlap(
        ms_ref=ms,
        ms_target=ms,
        ref_bands=[1, 2],
        pitch=0.5,
    )
    assert len(tracking["best_matches"]) == 2
    match1 = tracking["best_matches"][0]
    match2 = tracking["best_matches"][1]
    assert match1["ref_band"] == 1
    assert match1["target_band"] == 1
    assert pytest.approx(match1["max_overlap"], abs=1e-4) == 1.0
    assert pytest.approx(match1["delta_frequency"], abs=1e-6) == 0.0
    assert match1["ref_wavelength_nm"] is not None

    assert match2["ref_band"] == 2
    assert match2["target_band"] == 2
    assert pytest.approx(match2["max_overlap"], abs=1e-4) == 1.0

    assert len(tracking["subspace_projection"]) == 3
    assert tracking["ranked_target_bands"][:2] in ([1, 2], [2, 1])

    # 6. Error handling
    with pytest.raises(ValueError, match="Unknown field type"):
        compute_mode_overlap(ms, 1, ms, 1, field="invalid_field")

    with pytest.raises(ValueError, match="ref_bands sequence must not be empty"):
        track_modes_by_overlap(ms, ms, ref_bands=[])


def test_select_tracked_modes_strategies() -> None:
    """Verifies that cluster mode tracking enforces multiplet cohesion and avoids distant band jumping."""
    from phc_mpb.classification import (
        select_tracked_modes,
        select_tracked_modes_bipartite,
        select_tracked_modes_cluster,
        select_tracked_modes_greedy,
    )

    # Synthetic scenario reproducing the Eval 67 anomaly:
    # 3 reference modes (one singlet, two doublet)
    # Target bands: 15, 16, 17, 18, 19, 20, 21, 22
    target_bands = [15, 16, 17, 18, 19, 20, 21, 22]
    # Frequencies: bands 15..18 near 0.768, bands 19..20 near 0.793, bands 21..22 near 0.805
    target_freqs = [0.7681, 0.7684, 0.7686, 0.7691, 0.7930, 0.7932, 0.8045, 0.8048]

    # Overlap matrix: rows = ref modes 1..3, cols = target bands 15..22
    # Band 15 has strong singlet overlap (row 0: 0.91)
    # Band 17 has strong doublet overlap (row 1: 0.42, row 2: 0.17)
    # Band 16 has weak doublet overlap (row 2: 0.20)
    # Band 22 (distant mode at 0.8048) has strong doublet overlap (row 1: 0.25, row 2: 0.39 -> sum=0.64)
    overlap_mat = np.zeros((3, 8))
    overlap_mat[:, 0] = [0.91, 0.04, 0.01]  # band 15 (sum=0.96)
    overlap_mat[:, 1] = [0.05, 0.02, 0.20]  # band 16 (sum=0.27)
    overlap_mat[:, 2] = [0.15, 0.42, 0.17]  # band 17 (sum=0.74)
    overlap_mat[:, 3] = [0.02, 0.12, 0.15]  # band 18 (sum=0.29)
    overlap_mat[:, 4] = [0.00, 0.54, 0.01]  # band 19 (sum=0.55)
    overlap_mat[:, 5] = [0.00, 0.01, 0.54]  # band 20 (sum=0.55)
    overlap_mat[:, 6] = [0.00, 0.39, 0.25]  # band 21 (sum=0.64)
    overlap_mat[:, 7] = [0.00, 0.25, 0.39]  # band 22 (sum=0.64)

    # 1. Greedy strategy falls for Band 22 (reproducing the bug: [15, 17, 22])
    greedy_tracked, _greedy_ranked = select_tracked_modes_greedy(
        overlap_mat=overlap_mat, target_bands=target_bands, k_modes=3
    )
    assert 22 in greedy_tracked
    assert greedy_tracked == [15, 17, 22]

    # 2. Cluster strategy (Solution 4) enforces multiplet cohesion and selects [15, 16, 17]
    cluster_tracked, cluster_ranked = select_tracked_modes_cluster(
        overlap_mat=overlap_mat,
        target_bands=target_bands,
        k_modes=3,
        target_frequencies=target_freqs,
    )
    assert 22 not in cluster_tracked
    assert cluster_tracked == [15, 16, 17]
    assert cluster_ranked[:3] == [15, 16, 17]

    # 3. Bipartite strategy (Solution 2) with window constraint also selects cohesive cluster
    bipartite_tracked, _bipartite_ranked = select_tracked_modes_bipartite(
        overlap_mat=overlap_mat,
        target_bands=target_bands,
        k_modes=3,
        target_frequencies=target_freqs,
        cluster_window=4,
    )
    assert 22 not in bipartite_tracked
    assert set(bipartite_tracked).issubset({15, 16, 17, 18})

    # 4. Dispatcher verifies supported strategies and error handling
    disp_cluster, _ = select_tracked_modes(
        "cluster", overlap_mat, target_bands, 3, target_frequencies=target_freqs
    )
    assert disp_cluster == [15, 16, 17]

    disp_greedy, _ = select_tracked_modes(
        "greedy", overlap_mat, target_bands, 3, target_frequencies=target_freqs
    )
    assert disp_greedy == [15, 17, 22]

    with pytest.raises(ValueError, match="Unknown tracking strategy 'invalid'"):
        select_tracked_modes(
            "invalid",
            overlap_mat,
            target_bands,
            3,  # type: ignore[arg-type]
        )


def test_midplane_overlap_compatibility() -> None:
    """Tests 2D mid-plane spatial overlap extraction and cross-thickness field compatibility."""
    from phc_mpb.classification import (
        compute_mode_overlap_matrix,
        extract_midplane_field,
        track_modes_by_overlap,
    )

    # 1. Direct slice testing on extract_midplane_field
    assert extract_midplane_field(None) is None

    arr_5z = np.zeros((16, 16, 5, 3), dtype=complex)
    arr_5z[:, :, 2, :] = 1.0  # midplane slice
    sliced_5z = extract_midplane_field(arr_5z)
    assert sliced_5z is not None
    assert sliced_5z.shape == (16, 16, 3)
    assert np.all(sliced_5z == 1.0)

    arr_2d = np.ones((16, 16, 3), dtype=complex)
    sliced_2d = extract_midplane_field(arr_2d)
    assert sliced_2d is not None
    assert sliced_2d.shape == (16, 16, 3)

    # 2. Cross-thickness compatibility in compute_mode_overlap_matrix
    # Ref has 5 z-slices (e.g. h/a = 0.25), target has 3 z-slices (e.g. h/a = 0.20)
    rng = np.random.default_rng(42)
    ref_field_e = rng.standard_normal((16, 16, 5, 3)) + 1j * rng.standard_normal(
        (16, 16, 5, 3)
    )
    ref_field_d = rng.standard_normal((16, 16, 5, 3)) + 1j * rng.standard_normal(
        (16, 16, 5, 3)
    )
    tar_field_e = rng.standard_normal((16, 16, 3, 3)) + 1j * rng.standard_normal(
        (16, 16, 3, 3)
    )
    tar_field_d = rng.standard_normal((16, 16, 3, 3)) + 1j * rng.standard_normal(
        (16, 16, 3, 3)
    )

    ref_dict = {1: (ref_field_e, ref_field_d)}
    tar_dict = {1: (tar_field_e, tar_field_d)}

    # A. Under overlap_mode='midplane' (default):
    mat_midplane = compute_mode_overlap_matrix(
        ms_ref=ref_dict,
        ms_target=tar_dict,
        bands_ref=[1],
        bands_target=[1],
        overlap_mode="midplane",
    )
    assert mat_midplane.shape == (1, 1)
    assert 0.0 <= mat_midplane[0, 0] <= 1.0

    # Self-overlap of mid-plane slice must be 1.0
    mat_self = compute_mode_overlap_matrix(
        ms_ref=ref_dict,
        ms_target=ref_dict,
        bands_ref=[1],
        bands_target=[1],
        overlap_mode="midplane",
    )
    assert pytest.approx(mat_self[0, 0], abs=1e-5) == 1.0

    # B. Under overlap_mode='slab' with default interpolate=True: succeeds with resampling
    mat_slab_interp = compute_mode_overlap_matrix(
        ms_ref=ref_dict,
        ms_target=tar_dict,
        bands_ref=[1],
        bands_target=[1],
        overlap_mode="slab",
    )
    assert mat_slab_interp.shape == (1, 1)
    assert 0.0 <= mat_slab_interp[0, 0] <= 1.0

    # Under overlap_mode='slab' with interpolate=False: must raise ValueError due to 5 vs 3 z-slice mismatch
    with pytest.raises(ValueError, match="Field grid dimension mismatch"):
        compute_mode_overlap_matrix(
            ms_ref=ref_dict,
            ms_target=tar_dict,
            bands_ref=[1],
            bands_target=[1],
            overlap_mode="slab",
            interpolate=False,
        )

    # C. Invalid overlap_mode
    with pytest.raises(ValueError, match="Unknown overlap_mode 'unsupported'"):
        compute_mode_overlap_matrix(
            ms_ref=ref_dict,
            ms_target=tar_dict,
            bands_ref=[1],
            bands_target=[1],
            overlap_mode="unsupported",  # type: ignore[arg-type]
        )

    # D. track_modes_by_overlap forwards overlap_mode and interpolate
    tracking = track_modes_by_overlap(
        ms_ref=ref_dict,
        ms_target=tar_dict,
        ref_bands=[1],
        target_bands=[1],
        overlap_mode="midplane",
    )
    assert tracking["overlap_mode"] == "midplane"
    assert len(tracking["best_matches"]) == 1


def test_interpolate_field_to_grid() -> None:
    """Verifies grid interpolation of complex scalar and vector fields with periodic boundaries."""
    from phc_mpb.classification import (
        compute_mode_overlap,
        interpolate_field_to_grid,
    )

    rng = np.random.default_rng(123)

    # 1. Identity when shapes match
    arr_2d = rng.standard_normal((16, 16, 3)) + 1j * rng.standard_normal((16, 16, 3))
    arr_ident = interpolate_field_to_grid(arr_2d, (16, 16, 3))
    assert np.array_equal(arr_2d, arr_ident)

    # 2. Resampling 2D vector field (16, 16, 3) -> (32, 32, 3)
    # Smooth wave field: sin(2*pi*x/L)
    x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    y = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    smooth_e = np.zeros((16, 16, 3), dtype=complex)
    smooth_e[:, :, 2] = np.sin(xx) * np.cos(yy) + 1j * np.cos(xx) * np.sin(yy)
    smooth_d = smooth_e.copy()

    interp_e = interpolate_field_to_grid(smooth_e, (32, 32, 3))
    interp_d = interpolate_field_to_grid(smooth_d, (32, 32, 3))
    assert interp_e.shape == (32, 32, 3)

    # Compute overlap between 16x16 and 32x32: with interpolate=True it should be ~1.0
    ms_ref = {1: (smooth_e, smooth_d)}
    ms_tar = {1: (interp_e, interp_d)}
    overlap = compute_mode_overlap(ms_ref, 1, ms_tar, 1, interpolate=True)
    assert pytest.approx(overlap, abs=0.02) == 1.0

    # 3. 3D slab field resampling (16, 16, 5, 3) -> (24, 24, 7, 3)
    field_3d = rng.standard_normal((16, 16, 5, 3)) + 1j * rng.standard_normal(
        (16, 16, 5, 3)
    )
    interp_3d = interpolate_field_to_grid(field_3d, (24, 24, 7, 3))
    assert interp_3d.shape == (24, 24, 7, 3)

    # 4. Error on incompatible shapes
    with pytest.raises(ValueError, match="Cannot interpolate field of shape"):
        interpolate_field_to_grid(np.ones((16, 16)), (16, 16, 5, 3))
