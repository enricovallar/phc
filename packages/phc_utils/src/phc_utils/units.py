def normalized_freq_to_wavelength(
    norm_freq: float,
    pitch_um: float,
) -> float:
    """Converts MPB dimensionless normalized frequency to free-space wavelength in microns.

    In MPB, normalized frequency is defined as:
        omega_tilde = omega * a / (2 * pi * c) = a / lambda_0
    Therefore:
        lambda_0 = a / omega_tilde

    Args:
        norm_freq: Dimensionless frequency omega * a / (2 * pi * c).
        pitch_um: Lattice pitch constant a in micrometers.

    Returns:
        Free-space optical wavelength lambda_0 in micrometers.

    Raises:
        ValueError: If norm_freq <= 0 or pitch_um <= 0.
    """
    if norm_freq <= 0:
        raise ValueError(
            f"Normalized frequency must be strictly positive, got {norm_freq}."
        )
    if pitch_um <= 0:
        raise ValueError(f"Pitch must be strictly positive, got {pitch_um}.")
    return pitch_um / norm_freq


def wavelength_to_normalized_freq(
    wavelength_um: float,
    pitch_um: float,
) -> float:
    """Converts free-space wavelength in microns to MPB dimensionless frequency.

    Args:
        wavelength_um: Free-space optical wavelength lambda_0 in micrometers.
        pitch_um: Lattice pitch constant a in micrometers.

    Returns:
        Dimensionless frequency omega_tilde = a / lambda_0.

    Raises:
        ValueError: If wavelength_um <= 0 or pitch_um <= 0.
    """
    if wavelength_um <= 0:
        raise ValueError(f"Wavelength must be strictly positive, got {wavelength_um}.")
    if pitch_um <= 0:
        raise ValueError(f"Pitch must be strictly positive, got {pitch_um}.")
    return pitch_um / wavelength_um
