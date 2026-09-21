import pytest
from phc_utils.units import (
    normalized_freq_to_wavelength,
    wavelength_to_normalized_freq,
)


def test_normalized_freq_to_wavelength():
    # pitch = 0.45 um, norm_freq = 0.3 -> lambda = 0.45 / 0.3 = 1.5 um
    wl = normalized_freq_to_wavelength(0.3, 0.45)
    assert pytest.approx(wl, rel=1e-5) == 1.5


def test_wavelength_to_normalized_freq():
    # pitch = 0.45 um, lambda = 1.5 um -> norm_freq = 0.45 / 1.5 = 0.3
    nf = wavelength_to_normalized_freq(1.5, 0.45)
    assert pytest.approx(nf, rel=1e-5) == 0.3


def test_invalid_unit_inputs():
    with pytest.raises(ValueError):
        normalized_freq_to_wavelength(-0.1, 0.45)
    with pytest.raises(ValueError):
        wavelength_to_normalized_freq(1.5, -0.45)
