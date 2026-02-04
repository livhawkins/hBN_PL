import numpy as np
from pathlib import Path
from hbn_pl.SPE3reading import SPE3map
import matplotlib.pyplot as plt

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

