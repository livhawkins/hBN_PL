import numpy as np
from pathlib import Path
from hbn_pl.SPE3reading import SPE3map
import pandas as pd

def load_spe(path: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Load a .spe file and return the wavelength and spectral frames. Validate the path and data.
    
    Args:
        path (str or Path): Path to the .spe file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Wavelength array and multi dimensional frames array.
    '''
    if not path.exists():
        raise FileNotFoundError(f"SPE file not found: {path}")

    if path.suffix.lower() != ".spe":
        raise ValueError(f"Not a .spe file: {path}")
     
    spe = SPE3map(fname=path)
    wavelength = np.asarray(spe.wavelength)
    frames = np.asarray(spe.data[:, 0, :]) # Remove noise from extra channels

    if wavelength.ndim != 1:
        raise ValueError(f"Wavelength must be 1D, got shape {wavelength.shape}")

    if wavelength.size == 0:
        raise ValueError("Wavelength axis is empty")
    
    if frames.shape[0] == 0:
        raise ValueError("No frames found in SPE file")

    if not np.isfinite(frames).all():
        raise ValueError("Frames contain NaN or Inf values")

    if np.all(frames == 0):
        raise ValueError("All spectral values are zero (corrupt or empty file)")
    
    return wavelength, frames


def save_preprocess_results(spe_path: str, intensity: np.ndarray, energy: np.ndarray, fig, zpl_wavelength: float) -> None:
    """
    Save processed, cleaned energy spectrum, figure, and record ZPL wavelength into a common csv file in the folder.

    Args:
        spe_path : str
            Path to original .spe file.
        intensity : np.ndarray
            Intensity array.
        energy : np.ndarray
            Energy array.
        fig : matplotlib.figure.Figure
            Figure object to save.
        zpl_wavelength : float
            ZPL wavelength in nm.
    
    Returns:
        None: Saves .npz file with energy and intensity, figure as .png, and updates/creates CSV with ZPL wavelength.
    """
    spe_path = Path(spe_path)
    base_name = spe_path.stem # strip directory & extension
    folder = spe_path.parent

    npz_path = folder / f"{base_name}_cleaned.npz" #path for energy and intensity arrays
    png_path = folder / f"{base_name}_cleaned.png"
    csv_path = folder / "ZPL_wavelengths.csv"

    np.savez(npz_path, intensity=intensity, energy=energy, zpl_wavelength = zpl_wavelength, source_file = base_name) #save energy and intensity arrays as .npz file
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    #Update the csv file with the ZPL wavelength for this emitter. If the file doesn't exist, create it. If it exists, update the row for this emitter or add a new row if not present.
    new_row = pd.DataFrame({
        "filename": [base_name],
        "zpl_wavelength_nm": [zpl_wavelength]
    })

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        if base_name in df["filename"].values:
            df.loc[df["filename"] == base_name, "zpl_wavelength_nm"] = zpl_wavelength
        else:
            df = pd.concat([df, new_row], ignore_index=True)

    else:
        df = new_row

    df.to_csv(csv_path, index=False)

    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")
    print(f"Updated: {csv_path}")

