import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fit_phonon_peak(
    energy: np.ndarray,
    spectrum: np.ndarray,
    target: float,
    window: float = 8) -> dict:
    """
    Fit a phonon sideband peak with a Gaussian function.

    Parameters
    ----------
    energy : np.ndarray
        Energy relative to ZPL (meV).
    spectrum : np.ndarray
        Intensity array.
    target : float
        Expected phonon energy (meV).
    window : float
        Half-width of fitting window (meV).

    Returns
    -------
    dict with fitted parameters
    """

    def gaussian(x, A, x0, sigma, C):
        return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + C

    mask = (energy > target - window) & (energy < target + window)

    x_fit = energy[mask]
    y_fit = spectrum[mask]

    if len(x_fit) < 5:
        raise ValueError(f"Not enough data near {target} meV")

    # Initial guesses
    A0 = np.max(y_fit) - np.min(y_fit)
    x0 = target
    sigma0 = window / 3
    C0 = np.min(y_fit)

    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, x0, sigma0, C0])
    A, center, sigma, C = popt
    center_err = np.sqrt(np.diag(pcov))[1]

    return {
        "center": center,
        "center_err": center_err,
        "amplitude": A,
        "sigma": sigma,
        "background": C
    }