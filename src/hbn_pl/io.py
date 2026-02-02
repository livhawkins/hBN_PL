import numpy as np
from pathlib import Path

from hbn_pl.SPE3reading import SPE3map

def load_spe(path):
    # Input in the doc strings
    spe = SPE3map(fname=path)

    wavelength = np.asarray(spe.wavelength)

    # Is the SPE.data shape is usually (n_frames, n_channels, n_pixels)?
    # You were using [:, 0, :] to drop the extra channel - is this general?
    frames = np.asarray(spe.data[:, 0, :])

    return wavelength, frames


# It's also important that we evaluate things 
# and make sure the user isn't putting in jibberish 
# Can you go through these and check what it looks like?

def validate_spe_data(path, wavelength, frames):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"SPE file not found: {path}")

    if path.suffix.lower() != ".spe":
        raise ValueError(f"Not a .spe file: {path}")

    if wavelength.ndim != 1:
        raise ValueError(
            f"Wavelength must be 1D, got shape {wavelength.shape}"
        )

    if wavelength.size == 0:
        raise ValueError("Wavelength axis is empty")

    # Can we check the shape of the SPE file somehow? Is it always the same?

    if frames.shape[0] == 0:
        raise ValueError("No frames found in SPE file")

    if not np.isfinite(frames).all():
        raise ValueError("Frames contain NaN or Inf values")

    if np.all(frames == 0):
        raise ValueError("All spectral values are zero (corrupt or empty file)")
