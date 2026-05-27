from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_frames(frames: np.ndarray, wavelength: np.ndarray, show: bool = True) -> list[plt.Figure]:
    """
    Plot all spectral frames for visualization.

    Args:
        wavelength (np.ndarray): 1D array of wavelength values.
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
        show (bool): Whether to display the plots immediately. If False, returns a list of figure objects for later use.
    
    Returns:
        list[plt.Figure]: List of matplotlib figure objects for each frame.
    """
    plt.style.use('bmh')
    figs = []
    for i in range(frames.shape[0]):
        fig = plt.figure()
        plt.plot(wavelength, frames[i], color="#7A68A6")
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
    plt.style.use('bmh')
    for frame_idx, wl_list in cosmic_location.items():
        spectrum = frames[frame_idx]
        plt.plot(wavelength, spectrum, label="Spectrum", color="#7A68A6")

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


def plot_spectrum(x: np.ndarray, spectrum: np.ndarray, peaks=None, zpl=None, psb=None, x_quantity="Wavelength (nm)", outpath=None) -> None:
    '''
    Plot a single spectrum with optional peak, ZPL, and PSB markers.
    Args:
        spectrum (np.ndarray): 1D array of spectral intensity values.
        x (np.ndarray): 1D array of x-axis values. (wavelength or energy). Units are irrelevant as long as consistent with peaks, zpl, and psb. E.g. energy in meV, wavelength in nm etc.
        peaks (np.ndarray, optional): Indices of detected peaks in the spectrum.
        zpl (dict, optional): Dictionary with 'wl' and 'I' keys for ZPL marker.
        psb (dict, optional): Dictionary with 'wl' and 'I' keys for PSB marker.
        x_quantity (str): Label for the x-axis.
        outpath (str or Path, optional): If provided, save the plot to this path.

    Returns:
        None
    ''' 
    plt.style.use('bmh')
    plt.figure()
    plt.plot(x, spectrum, color="#7A68A6", label="Spectrum")

    if peaks is not None and len(peaks) > 0:
        plt.plot(
            x[peaks],
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

    plt.xlabel(x_quantity)
    plt.ylabel("PL intensity")
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_energy(wavelength: np.ndarray, spectrum: np.ndarray, zpl_wavelength: float) -> tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Plot energy spectrum centred on the ZPL.

    Args:
        wavelength (np.ndarray): 1D array of wavelength values.
        spectrum (np.ndarray): 1D array of spectral intensity values.
        zpl_wavelength (float): Wavelength of the ZPL to centre the plot.

    Returns:
        tuple[np.ndarray, np.ndarray, plt.Figure]: Tuple containing the intensity and corresponding energy values, and the matplotlib figure object to save later.
    """
    plt.style.use('bmh')
    energy = 1239.84 / wavelength  # Convert wavelength to energy in eV for wavelength in nm units
    zpl_energy = 1239.84 / zpl_wavelength
    energy_offset = -1000*(energy - zpl_energy) #shift to ZPL = 0 eV and convert to meV
    mask = (energy_offset >= -20) & (energy_offset <= 200) #only really interested in -20 to 200 meV range for PSB

    fig, ax = plt.subplots()

    ax.plot(energy_offset[mask], spectrum[mask], color="#7A68A6")
    ax.set_xlabel("Phonon energy (meV)")
    ax.set_ylabel("Normalised PL Intensity")

    fig.tight_layout()

    return spectrum[mask], energy_offset[mask], fig


def plot_psb_plotly(energy: np.ndarray, spectrum: np.ndarray, fit_results: list, targets: list, window: float = 8, filename: str = "psb_interactive.html") -> None:
    '''
    Plot the phonon sideband spectrum with interactive Plotly, overlaying Gaussian fits for each target phonon energy.
    Args:
        energy (np.ndarray): 1D array of energy values (meV).
        spectrum (np.ndarray): 1D array of spectral intensity values.
        fit_results (list): List of dictionaries containing fitted parameters for each phonon peak.
        targets (list): List of target phonon energies corresponding to the fit results.
        window (float): Half-width of fitting window (meV) for plotting the Gaussian fits.
        filename (str): Filename to save the interactive plot as HTML.
    
    Returns:
        None
    '''
    fit_colors = [
        "#D62728",  # red
        "#1F77B4",  # blue
        "#2CA02C",  # green
        "#9467BD"   # purple
    ]

    fig = go.Figure()
    fig.update_layout(template="simple_white")

    # Plot full spectrum
    fig.add_trace(go.Scatter(
        x=energy,
        y=spectrum,
        mode='lines',
        name='Spectrum',
        line=dict(color='black')
    ))

    # Overlay fitted Gaussians
    for i, (res, target) in enumerate(zip(fit_results, targets)):

        A = res['amplitude']
        x0 = res['center']
        sigma = res['sigma']
        C = res['background']

        mask = (energy >= x0 - window) & (energy <= x0 + window)
        x_dense = energy[mask]
        y_dense = A * np.exp(-(x_dense - x0)**2 / (2*sigma**2)) + C

        # Define colour of gaussian fit
        color = fit_colors[i % len(fit_colors)]

        # Gaussian line
        fig.add_trace(go.Scatter(
            x=x_dense,
            y=y_dense,
            mode='lines',
            name=f"{target} meV fit",
            line=dict(color=color, width=3)
        ))
        
        # Marker at center
        fig.add_trace(go.Scatter(
            x=[x0],
            y=[A + C],
            mode='markers',
            marker=dict(color=color, size=10, symbol='x'),
            showlegend=False
        ))
    
    title_text = Path(filename).stem.replace("_interactive", "")
    
    fig.update_layout(
        title=dict(
        text=title_text,
        x=0.5,              # center title
        xanchor="center"
    ),
        xaxis_title="Phonon energy (meV)",
        yaxis_title="Normalised Intensity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.write_html(filename)
    print(f"Interactive figure saved to {filename}")
    fig.show()