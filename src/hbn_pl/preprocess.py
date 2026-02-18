import numpy as np
from scipy.signal import find_peaks, peak_widths
import spectrapepper as spep
import matplotlib.pyplot as plt

#order of preprocessing steps:
#1. background subtraction
#2. create normalised copy of frames for cosmic ray detection
#3. detect cosmic ray frames using normalised frames
#4. remove cosmic rays from un-normalised frames
#5. remove bad frames from un-normalised frames 
# (now no cosmic rays so easier for median comparison) 
# - NOT on normalised frames because we need to detect when emitter dies/blinks
#6. average cleaned frames
#7. normalise averaged spectrum

def detect_bad_frames_simple(frames: np.ndarray, drop_fraction: float) -> list[int]:
    '''
    Simple detection of bad frames where overall intensity drops below a fraction of the median intensity.
    Note that the use of median makes this robust to a few bad frames, but if the majority of frames are bad
    this method may fail. In that case, use detect_bad_frames function with local windowing.
    Alternatively could compare SNR, but since the noise is very similar between frames, comparing signal intensity is sufficient.
    Args:
        frames (np.ndarray): 2D array of spectral frames
                             (num_frames x num_wavelengths). 
        drop_fraction (float): Fractional drop threshold below the median intensity when a frame is flagged.
                               e.g. 0.5 flags frames with >50% drop.
    Returns:
        list[int]: Indices of frames considered bad.
    '''
    print("Running simple bad frame detection...")
    frame_intensity = np.sum(frames, axis=1)
    median = np.median(frame_intensity)
    bad_frames = np.where(frame_intensity < drop_fraction * median)[0] # Flag frames below drop_fraction of of median
    print(f"Detected bad frames due to intensity drop {drop_fraction}: {bad_frames}")
    return bad_frames
    
def detect_bad_frames_complex(frames: np.ndarray, window: int, drop_fraction: float) -> list[int]:
    """
    Detect frames where the overall signal drops significantly
    relative to neighbouring frames via median comparison e.g. from blinking, photobleaching etc.
    Uses rolling comparison over a local window of frames to account for many bad frames.
    detect_bad_frames_simple should be used first, and resort to this function if the majority of frames are bad.
    Args:
        frames (np.ndarray): 2D array of spectral frames
                             (num_frames x num_wavelengths). 
        window (int): Number of neighbouring frames to consider on each side. 
        drop_fraction (float): Fractional drop threshold below the reference intensity when a frame is flagged.
                               e.g. 0.5 flags frames with >50% drop.
    Returns:
        list[int]: Indices of frames considered bad.
    """
    print("Running complex bad frame detection...")
    n_frames = frames.shape[0]
    frame_intensity = np.sum(frames, axis=1)
    bad_frames = []

    for i in range(n_frames):
        # Define local window
        lo = max(0, i - window)
        hi = min(n_frames, i + window + 1)
        local_frames = np.delete(frame_intensity[lo:hi], i - lo)  # exclude self

        if len(local_frames) == 0:
            continue

        local_ref = np.median(local_frames)

        if frame_intensity[i] < drop_fraction * local_ref:
            bad_frames.append(i)

    print(f"Detected bad frames due to intensity drop {drop_fraction}: {bad_frames}")
    return bad_frames
          


def remove_frames(frames: np.ndarray, frames_to_remove: list[int]) -> np.ndarray:
    '''
    Remove specified frames from the dataset.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
        frames_to_remove (list[int]): List of frame indices to remove.
    Returns:
        np.ndarray: Frames with specified frames removed.
    '''
    print(f"Removing {len(frames_to_remove)} frames: {frames_to_remove}")
    if frames_to_remove is None or len(frames_to_remove) == 0:
        return frames
    deleted = np.delete(frames, frames_to_remove, axis=0)
    print("Successfully removed bad frames.")
    return deleted


def is_cosmic_by_difference(frames: np.ndarray, frame_idx: int, peak_pix: int, z_thresh: float, half_width: int, noise_width: int) -> bool:
    '''
    Secondary check to verify if a narrow peak identified in detect_cosmic_frames is likely a cosmic ray by comparing to neighbouring frames.
    Args:
        frames (np.ndarray): shape (n_frames, n_pixels), temporarily normalised spectra
        frame_idx (int): index of the frame being checked
        peak_pix (int): pixel index of the candidate peak in this frame
        z_thresh (float): z-score threshold for confirming cosmic ray; higher = more conservative
        half_width (int): number of pixels on either side of the peak to consider for local peak amplitude
        noise_width (int): number of pixels on either side of the peak to consider for local noise estimation (excluding peak region)

    Returns:
        bool: True if the peak is likely a cosmic ray, False otherwise
    
    '''
    if frame_idx == 0:
        neighbour = frames[frame_idx + 1]
    elif frame_idx == frames.shape[0] - 1:
        neighbour = frames[frame_idx - 1]
    else:
        neighbour = 0.5 * (frames[frame_idx - 1] + frames[frame_idx + 1])

    diff = frames[frame_idx] - neighbour

    # defining local peak window
    lo = max(0, peak_pix - half_width)
    hi = min(len(diff), peak_pix + half_width + 1)
    local_peak = diff[lo:hi]

    peak_amp = np.max(np.abs(local_peak))

    # defining noise region around but excluding the peak
    nlo = max(0, peak_pix - noise_width)
    nhi = min(len(diff), peak_pix + noise_width + 1)

    noise_region = np.concatenate([
        diff[nlo:lo],
        diff[hi:nhi]
    ])

    if len(noise_region) < 10:
        return False

    mad = np.median(np.abs(noise_region - np.median(noise_region)))
    sigma_local = 1.4826 * mad #local noise estimate using MAD

    if sigma_local == 0:
        return False

    z = peak_amp / sigma_local

    return z > z_thresh


def detect_cosmic_frames(frames: np.ndarray, wavelength: np.ndarray, prominence_threshold: float, fwhm_threshold: float, 
                         n_peaks: int, z_thresh: float, half_width: int, noise_width: int) -> tuple[list[int], dict[int, list[float]]]:
    '''
    Method to detect cosmic ray frames by looking for very narrow peaks in the spectrum. Verify these are cosmic rays and 
    not real ZPLs by passing through a second check comparing to neighbouring frames (is_cosmic_by_difference).
    Args:        
        frames (np.ndarray): shape (n_frames, n_pixels), temporarily normalised spectra
        wavelength (np.ndarray): 1D array of wavelength values corresponding to the spectral pixels
        prominence_threshold (float): minimum prominence for peak detection; higher = fewer peaks detected. 
            Some cosmic rays can be faint so this should not be too high.
        fwhm_threshold (float): maximum FWHM for a peak to be considered a cosmic ray; lower = more aggressive detection
        n_peaks (int): number of top peaks to consider in each frame; higher = more peaks checked for cosmic ray criterion
        z_thresh (float): z-score threshold for confirming cosmic ray in second check; higher = more conservative
        half_width (int): number of pixels on either side of the peak to consider for local peak amplitude in second check
        noise_width (int): number of pixels on either side of the peak to consider for local noise estimation in second check
    
    Returns:
        list[int]: indices of frames likely containing cosmic rays
        dict[int, list[float]]: mapping of frame index (int) to list of wavelengths where cosmic rays were detected
    '''
    print("Detecting cosmic ray frames...")
    cosmic_frames = []
    cosmic_location = {}  # frame_idx -> list of wavelengths
    n_frames = frames.shape[0]

    for frame_idx in range(n_frames):
        spectrum = frames[frame_idx]

        # Find peaks in this frame
        peaks, properties = find_peaks(spectrum,prominence=prominence_threshold)

        if len(peaks) == 0:
            continue

        prominences = properties["prominences"]
        top_peaks = peaks[np.argsort(prominences)[::-1][:n_peaks]] #sort in order of highest prominence to lowest

        # Compute FWHM for selected peaks
        fwhm_values, _, _, _ = peak_widths(spectrum,top_peaks,rel_height=0.5)

        candidate_peaks = top_peaks[fwhm_values < fwhm_threshold] #find some candidate peaks for second check

        confirmed_cosmics = []

        for peaks in candidate_peaks:
            if is_cosmic_by_difference(frames, frame_idx, peaks, z_thresh=z_thresh, half_width=half_width, noise_width=noise_width):#returns true if second check believes it is a cosmic ray
                confirmed_cosmics.append(peaks)

        if confirmed_cosmics:
            cosmic_frames.append(frame_idx)
            cosmic_location[frame_idx] = wavelength[confirmed_cosmics].tolist()
    print(f"Detected {len(cosmic_frames)} cosmic ray frames: {cosmic_frames}")
    print(f"Cosmic ray wavelengths: {cosmic_location}")
    print("Cosmic ray detection complete.")
    return cosmic_frames, cosmic_location



def remove_cosmic_rays(frames: np.ndarray, cosmic_frames: list[int], sigma: float) -> np.ndarray:
    """
    Automatically remove cosmic rays from spectral frames using spectrapepper's cosmicmed function.

    Args:
        frames (np.ndarray): background-corrected, (normalised) frames
        cosmic_frames (list[int]): indices of frames likely containing cosmic rays
        sigma (float): sigma parameter for spectrapepper.cosmicmed; lower sigma = more aggressive correction

    Returns:
        np.ndarray: cleaned frames
    """
    print(f"Removing cosmic rays from {len(cosmic_frames)} frames: {cosmic_frames}")
    if len(cosmic_frames) == 0:
        return frames

    frames_clean = frames.copy()
    n_frames = frames.shape[0]

    for i in cosmic_frames:
        if i <= 0 or i >= n_frames - 1:
            continue      # Cannot apply median method at boundaries

        spectra_list = [
            frames[i - 1],
            frames[i],
            frames[i + 1],
        ]

        corrected = spep.cosmicmed(spectra_list, sigma=sigma)
        frames_clean[i] = corrected[1]

    for i in cosmic_frames:
        plt.figure()
        plt.plot(frames[i], label="Original")
        plt.plot(frames_clean[i], label="Corrected")
        plt.legend()
        plt.title(f"Frame {i}")
        plt.show()

    print("Successfully removed cosmic rays.")
    return frames_clean

def remove_cosmic_rays2(frames, wavelength, cosmic_frames, cosmic_location,
                        sigma=2.5, half_width=3, show=True):
    frames_clean = frames.copy()
    cosmic_figs = []  # <-- define the list here

    for frame_idx in cosmic_frames:
        # run cosmic ray correction as before
        if frame_idx == 0:
            spectra_list = [frames[frame_idx], frames[frame_idx + 1], frames[frame_idx + 2]]
        elif frame_idx == frames.shape[0] - 1:
            spectra_list = [frames[frame_idx - 2], frames[frame_idx - 1], frames[frame_idx]]
        else:
            spectra_list = [frames[frame_idx - 1], frames[frame_idx], frames[frame_idx + 1]]

        corrected = spep.cosmicmed(spectra_list, sigma=sigma)
        corrected_spectrum = corrected[1]

        for wl in cosmic_location.get(frame_idx, []):
            mask = np.abs(wavelength - wl) <= half_width
            frames_clean[frame_idx, mask] = corrected_spectrum[mask]

        # Plot comparison
        fig, ax = plt.subplots()
        ax.plot(wavelength, frames[frame_idx], label="Original", alpha=0.7)
        ax.plot(wavelength, frames_clean[frame_idx], label="Corrected", alpha=0.7)
        ax.legend()
        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")

        cosmic_figs.append(fig)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return frames_clean, cosmic_figs


def background_subtract(frames: np.ndarray, bg_slice: tuple[int, int]) -> np.ndarray:
    '''
    Subtract background from each frame using the mean of a specified wavelength slice.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
        bg_slice (tuple): Tuple specifying the start and stop indices for background calculation.

    Returns:
        np.ndarray: Background-subtracted frames.
    '''
    print("Performing background subtraction...")
    start, stop = bg_slice
    background = np.mean(frames[:, start:stop], axis=1)
    corrected = frames - background[:, None]
    print("Successfully subtracted background.")
    return corrected


def normalise(frames: np.ndarray) -> np.ndarray:
    '''
    Normalise each frame by its maximum value.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).

    Returns:
        np.ndarray: Normalised frames.
    '''
    print("Normalising frames...")
    max_vals = np.max(frames, axis=1)
    frames_norm = frames / (max_vals[:, None] + 1e-20)
    print("Successfully normalised frames.")
    return frames_norm


def average_and_normalise(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Average all frames and normalise the result.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).

    Result:
        tuple[np.ndarray, np.ndarray]: Averaged spectrum and normalised averaged spectrum.
    '''
    print("Averaging and normalising frames into a single spectrum...")
    avg = np.mean(frames, axis=0)
    max_val = np.max(avg)
    avg_norm = avg / (max_val + 1e-20)
    print("Successfully averaged and normalised frames.")   
    return avg, avg_norm