import numpy as np
from scipy.signal import find_peaks


def find_peaks_pl(
    wavelength,
    spectrum,
    height=0.1,
    prominence=0.7,
):
    peaks, properties = find_peaks(
        spectrum,
        height=height,
        prominence=prominence,
    )
    return peaks, properties


def identify_zpl_psb(
    peaks,
    wavelength,
    spectrum,
    zpl_window=(610, 641),
    psb_min_wavelength=665,
):
    # docs please
    if len(peaks) == 0:
        return None, None

    peak_wl = wavelength[peaks]
    peak_I = spectrum[peaks]

    zpl_mask = (peak_wl > zpl_window[0]) & (peak_wl < zpl_window[1])
    zpl = None
    if np.any(zpl_mask):
        idx = np.argmax(peak_I[zpl_mask])
        zpl = {
            "wl": float(peak_wl[zpl_mask][idx]),
            "I": float(peak_I[zpl_mask][idx]),
        }

    psb_mask = peak_wl > psb_min_wavelength
    psb = None
    if np.any(psb_mask):
        idx = np.argmax(peak_I[psb_mask])
        psb = {
            "wl": float(peak_wl[psb_mask][idx]),
            "I": float(peak_I[psb_mask][idx]),
        }

    return zpl, psb


def classify_emitter(peaks, wavelength, zpl_window=(610, 641)):
    # Docs plz
    if len(peaks) == 0:
        return "uncertain"

    peak_wl = wavelength[peaks]
    zpl_peaks = peak_wl[
        (peak_wl > zpl_window[0]) & (peak_wl < zpl_window[1])
    ]

    n = len(zpl_peaks)

    if n == 1:
        return "1_ZPL"
    elif n == 2:
        return "2_ZPL"
    elif n >= 3:
        return "3_ZPL"
    else:
        return "uncertain"
