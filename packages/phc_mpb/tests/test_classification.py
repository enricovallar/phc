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
