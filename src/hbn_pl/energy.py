import numpy as np


def wavelength_to_energy_ev(wavelength_nm):

    return 1239.8 / wavelength_nm


def energy_offset_mev(wavelength_nm, zpl_wavelength_nm):

    energy = wavelength_to_energy_ev(wavelength_nm)
    zpl_energy = 1239.8 / zpl_wavelength_nm

    # Negative sign so PSB is positive meV
    return -1000.0 * (energy - zpl_energy)
