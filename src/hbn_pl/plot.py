import matplotlib.pyplot as plt

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


def plot_energy(
    energy_mev,
    spectrum,
    energy_window=(-20, 200),
    outpath=None,
):

    mask = (energy_mev >= energy_window[0]) & (
        energy_mev <= energy_window[1]
    )

    fig = plt.figure()
    plt.plot(energy_mev[mask], spectrum[mask])
    plt.xlabel("Phonon energy (meV)")
    plt.ylabel("Normalised PL intensity")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        return fig
