"""Unit tests for modal data filtering and alpha ramping in phc_mpb.filtering."""

import numpy as np
import pytest
from phc_mpb.classification import ModeClassification
from phc_mpb.filtering import (
    compute_confinement_alphas,
    create_confinement_mask,
    filter_band_data,
    filter_by_classification,
    filter_light_cone,
)


def test_create_confinement_mask():
    """Verifies boolean confinement mask thresholding."""
    confinements = np.array([[0.05, 0.20, 0.50], [0.19, 0.21, 0.95]])
    mask = create_confinement_mask(confinements, cutoff=0.20)

    expected = np.array([[False, True, True], [False, True, True]])
    np.testing.assert_array_equal(mask, expected)


def test_compute_confinement_alphas():
    """Verifies continuous alpha fading across cutoff and full thresholds."""
    confinements = np.array([0.10, 0.20, 0.35, 0.50, 0.90])
    alphas = compute_confinement_alphas(
        confinements, cutoff=0.20, full_threshold=0.50, base_alpha=1.0
    )

    # eta < 0.20 -> 0.0
    assert alphas[0] == 0.0
    # eta == 0.20 -> 0.0
    assert alphas[1] == 0.0
    # eta == 0.35 -> (0.35 - 0.20) / (0.50 - 0.20) = 0.15 / 0.30 = 0.50
    assert pytest.approx(alphas[2], 0.01) == 0.50
    # eta == 0.50 -> 1.0
    assert alphas[3] == 1.0
    # eta == 0.90 -> 1.0
    assert alphas[4] == 1.0


def test_filter_band_data():
    """Verifies artifact modes below confinement cutoff are replaced with NaN."""
    freqs = np.array([[0.2, 0.4], [0.3, 0.5]])
    confinements = np.array([[0.8, 0.1], [0.15, 0.9]])
    te_fracs = np.array([[0.9, 0.1], [0.2, 0.8]])

    filt_freqs, filt_fracs = filter_band_data(
        freqs, confinements, cutoff=0.20, te_fractions=te_fracs
    )

    assert not np.isnan(filt_freqs[0, 0])
    assert np.isnan(filt_freqs[0, 1])
    assert np.isnan(filt_freqs[1, 0])
    assert not np.isnan(filt_freqs[1, 1])

    assert filt_fracs is not None
    assert not np.isnan(filt_fracs[0, 0])
    assert np.isnan(filt_fracs[0, 1])
    assert np.isnan(filt_fracs[1, 0])
    assert not np.isnan(filt_fracs[1, 1])


def test_filter_by_classification():
    """Verifies filtering by physical mode classification category."""
    freqs = np.array([[0.2, 0.4, 0.6]])
    classes = np.array(
        [
            [
                ModeClassification.GUIDED.value,
                ModeClassification.LEAKY.value,
                ModeClassification.RADIATION.value,
            ]
        ]
    )

    # Keep guided + leaky, drop radiation (default)
    f_res = filter_by_classification(
        freqs, classes, keep_guided=True, keep_leaky=True, keep_radiation=False
    )
    assert not np.isnan(f_res[0, 0])
    assert not np.isnan(f_res[0, 1])
    assert np.isnan(f_res[0, 2])

    # Keep only guided
    f_guided = filter_by_classification(
        freqs, classes, keep_guided=True, keep_leaky=False, keep_radiation=False
    )
    assert not np.isnan(f_guided[0, 0])
    assert np.isnan(f_guided[0, 1])
    assert np.isnan(f_guided[0, 2])


def test_filter_light_cone():
    """Verifies light cone filtering."""
    freqs = np.array([[0.2, 0.4], [0.3, 0.5]])
    light_line = [0.3, 0.4]

    filtered = filter_light_cone(freqs, light_line)
    # k0: light_line=0.3. b0=0.2 (below, kept), b1=0.4 (above, NaN)
    assert not np.isnan(filtered[0, 0])
    assert np.isnan(filtered[0, 1])
    # k1: light_line=0.4. b0=0.3 (below, kept), b1=0.5 (above, NaN)
    assert not np.isnan(filtered[1, 0])
    assert np.isnan(filtered[1, 1])
