import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import peak_widths

def find_peaks(y: np.ndarray, x: np.ndarray = None, center_guess: float = None, window: float = None, height=None, width=None, prominence=None, distance=None,) -> tuple[np.ndarray, dict]:
    """
    Wrapper around scipy.signal.find_peaks with optional windowing.

    Args:
        y : np.ndarray
            1D array of PL intensity values. Units are irrelevant as long as consistent (i.e. nm, meV etc).
        x : np.ndarray, optional
            1D array of x-axis values (e.g. wavelength). Required if using center_guess and window.
        center_guess : float, optional
            Center position around which to search for peaks (in x-units).
        window : float, optional
            Half-width of the search window around center_guess (same units as x).
        height : (float or tuple, optional) 
            Required height of peaks. See scipy docs for details. 
        width : (float or tuple, optional)
            Required width of peaks. See scipy docs for details. 
        prominence : (float or tuple, optional)
            Required prominence of peaks. See scipy docs for details. 
        distance : (float, optional)
            Required minimum horizontal distance between peaks. See scipy docs for details.

    Returns:
        peaks : np.ndarray
            Indices of detected peaks (relative to original array).
        properties : dict
            Peak properties from scipy.
    """

    # If windowed search is requested
    if center_guess is not None and window is not None:
        if x is None:
            raise ValueError("x must be provided when using center_guess and window")

        mask = (x >= center_guess - window) & (x <= center_guess + window)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise ValueError("No data points inside specified window")

        peaks_local, properties = scipy_find_peaks(
            y[mask],
            height=height,
            width=width,
            prominence=prominence,
            distance=distance,
        )

        # Convert local indices back to global indices
        peaks = indices[peaks_local]

    else:
        peaks, properties = scipy_find_peaks(
            y,
            height=height,
            width=width,
            prominence=prominence,
            distance=distance,
        )

    return peaks, properties


def extract_peak_parameters(x: np.ndarray, y: np.ndarray, peaks: np.ndarray, properties: dict) -> list[dict]:
    """
    Calculate peak parameters including FWHM and return structured info.

    Args:
        x : np.ndarray
            1D array of x-axis values.
        y : np.ndarray
            1D array of signal values.
        peaks : np.ndarray
            Indices of detected peaks.
        properties : dict
            Properties dictionary returned by scipy.signal.find_peaks.

    Returns:
        list of dict
            Each dict contains:
            - location
            - intensity
            - fwhm
            - prominence
    """

    if peaks is None or len(peaks) == 0:
        raise ValueError("No peaks provided")

    # FWHM calculation
    widths, width_heights, left_ips, right_ips = peak_widths(
        y,
        peaks,
        rel_height=0.5
    )

    peak_list = []

    for i, peak_index in enumerate(peaks):

        fwhm = (
            x[int(right_ips[i])] -
            x[int(left_ips[i])]
        )

        peak_dict = {
            "location": x[peak_index],
            "intensity": y[peak_index],
            "fwhm": fwhm,
            "prominence": properties.get("prominences", [None]*len(peaks))[i]
        }

        peak_list.append(peak_dict)

    return peak_list


def correct_spectrum(x: np.ndarray, spectrum: np.ndarray, peak_params: list[dict], plot: bool = True) -> np.ndarray:
    """
    Remove weaker spectrally shifted ZPL replicas assuming identical lineshape.

    Args:
        x : np.ndarray
            x-axis values (e.g. wavelength).
        spectrum : np.ndarray
            Total spectrum.
        peak_params : list of dict
            Output from extract_peak_parameters() with the ZPL candidates centered around tall ZPL wavelength.
        plot : bool
            If True, show diagnostic plot.

    Returns:
        np.ndarray
            Corrected spectrum.
    """
    print("Performing multi-ZPL correction...")
    total_spectrum = spectrum.copy()
    corrected_spectrum = spectrum.copy()

    if len(peak_params) < 2:
        raise ValueError("Need at least two ZPLs for correction.")

    # Sort by intensity (ascending)
    peaks_sorted = sorted(peak_params, key=lambda p: p["intensity"])

    # Strongest ZPL = last element
    reference_peak = peaks_sorted[-1]
    lambda_ref = reference_peak["location"]
    I_ref = reference_peak["intensity"]

    cumulative_subtraction = np.zeros_like(spectrum)

    # Loop over all weaker peaks
    for peak in peaks_sorted[:-1]:

        lambda_i = peak["location"]
        I_i = peak["intensity"]

        weight = I_i / I_ref
        shift = lambda_i - lambda_ref

        shifted = weight * np.interp(
            x,
            x + shift,
            total_spectrum,
            left=0,
            right=0
        )

        cumulative_subtraction += shifted

    corrected_spectrum = total_spectrum - cumulative_subtraction
    corrected_spectrum = np.clip(corrected_spectrum, 0, None)
    print("Correction complete.")
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(x, total_spectrum, label="Original Spectrum")
        plt.plot(x, cumulative_subtraction, label="Total Subtracted Contribution")
        plt.plot(x, corrected_spectrum, label="Corrected Spectrum")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalised Intensity")
        plt.legend()
        plt.title("Multi-ZPL Removal")
        plt.show()

    return corrected_spectrum


def fit_peak_gaussian(x: np.ndarray, spectrum: np.ndarray, center_guess: float, window: float = 1.0) -> dict:
    """
    Fit a single peak with a Gaussian function within a specified window.

    Args:
        x : np.ndarray
            1D array of x-axis values. Units are irrelevant as long as consistent with center_guess and window. E.g. energy in meV, wavelength in nm etc.
        spectrum : np.ndarray
            1D array of corresponding y values. PL spectra can be in arbitrary units as long as consistent.
        center_guess : float
            Initial guess for peak center (in same units as x).
        window : float, optional
            Half-width of the fitting window around center_guess (same units as x).

    Returns:
        dict
            Dictionary containing fitted parameters:
            - center
            - center_err
            - amplitude
            - sigma
            - background
    """

    def gaussian(x, amplitude, center, sigma, background):
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + background

    # Select fitting window
    mask = (x > center_guess - window) & (x < center_guess + window)
    x_fit = x[mask]
    y_fit = spectrum[mask]

    if len(x_fit) < 5:
        raise ValueError(f"Not enough data near x = {center_guess}")

    # Initial parameter guesses
    amplitude0 = np.max(y_fit) - np.min(y_fit)
    sigma0 = window / 3
    background0 = np.min(y_fit)

    p0 = [amplitude0, center_guess, sigma0, background0]

    popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=p0)
    amplitude, center, sigma, background = popt

    center_err = np.sqrt(np.diag(pcov))[1]

    return {
        "center": center,
        "center_err": center_err,
        "amplitude": amplitude,
        "sigma": sigma,
        "background": background,
    }