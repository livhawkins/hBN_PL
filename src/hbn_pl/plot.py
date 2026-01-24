import matplotlib.pyplot as plt


def plot_spectrum(
    wavelength,
    spectrum,
    peaks=None,
    zpl=None,
    psb=None,
    outpath=None,
):

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

    plt.figure()
    plt.plot(energy_mev[mask], spectrum[mask])
    plt.xlabel("Phonon energy (meV)")
    plt.ylabel("Normalised PL intensity")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()
