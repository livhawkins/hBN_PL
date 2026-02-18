import matplotlib.pyplot as plt
import numpy as np

def plot_frames(wavelength, frames, show = True):
    """
    Plot all spectral frames for visualization.

    Args:
        wavelength (np.ndarray): 1D array of wavelength values.
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
    """
    figs = []
    for i in range(frames.shape[0]):
        fig = plt.figure()
        plt.plot(wavelength, frames[i])
        plt.title(f"Frame {i}")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (counts)')
        figs.append(fig)
        if show:
            plt.show()
    return figs

def plot_cosmic_frames(frames: np.ndarray, wavelength: np.ndarray, cosmic_location: dict[int, list[float]]) -> None:
    """
    Plot frames identified as containing cosmic rays with markers for the detected cosmic ray locations.

    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
        wavelength (np.ndarray): 1D array of wavelength values.
        cosmic_location (dict): Dictionary mapping frame indices to lists of detected cosmic ray wavelengths.
    """
    for frame_idx, wl_list in cosmic_location.items():
        spectrum = frames[frame_idx]
        plt.plot(wavelength, spectrum, label="Spectrum")

        for wl in wl_list:
            pix = np.argmin(np.abs(wavelength - wl))
            plt.plot(
                wavelength[pix],
                spectrum[pix],
                marker="x",
                color="red",
                markersize=10,
                mew=2,
                label="Detected cosmic ray"
            )

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title(f"Cosmic Ray Frame {frame_idx}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_spectrum(wavelength, spectrum, peaks=None, zpl=None, psb=None, outpath=None) -> None:
    '''
    Plot a single spectrum with optional peak, ZPL, and PSB markers.
    Args:
        wavelength (np.ndarray): 1D array of wavelength values.
        spectrum (np.ndarray): 1D array of spectral intensity values.
        peaks (np.ndarray, optional): Indices of detected peaks in the spectrum.
        zpl (dict, optional): Dictionary with 'wl' and 'I' keys for ZPL marker.
        psb (dict, optional): Dictionary with 'wl' and 'I' keys for PSB marker.
        outpath (str or Path, optional): If provided, save the plot to this path.

    Returns:
        None
    ''' 
    plt.figure()
    plt.plot(wavelength, spectrum, label="Spectrum")

    if peaks is not None and len(peaks) > 0:
        plt.plot(
            wavelength[peaks],
            spectrum[peaks],
            "x",
            color="red",
            label="Peaks",
        )

    if zpl is not None:
        plt.plot(
            zpl["wl"],
            zpl["I"],
            "x",
            color="green",
            label="ZPL",
            markersize=10,
        )

    if psb is not None:
        plt.plot(
            psb["wl"],
            psb["I"],
            "x",
            color="green",
            markersize=10,
            label="PSB",
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalised PL intensity")
    plt.legend()
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()
