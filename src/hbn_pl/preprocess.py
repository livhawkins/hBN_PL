import numpy as np
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
    if frames_to_remove is None or len(frames_to_remove) == 0:
        return frames

    return np.delete(frames, frames_to_remove, axis=0)


def detect_cosmic_frames(frames: np.ndarray, sigma_threshold: float, min_outliers: int) -> list[int]:
    """
    Automatically detect frames containing cosmic rays using neighbour comparison.

    Args:
        frames (np.ndarray): shape (n_frames, n_pixels), temporarily normalised spectra
        detect_sigma (float): detection threshold for cosmic rays; higher = fewer detections
        min_outliers (int): minimum number of spiky pixels to flag a frame as containing cosmic rays

    Returns:
        list[int]: indices of frames likely containing cosmic rays
    """
    cosmic_frames = []
    n_frames = frames.shape[0]

    for i in range(1, n_frames - 1):
        neighbour_median = np.median([frames[i - 1], frames[i + 1]], axis=0) #make a clean reference spectrum from
        #the median of neighbouring frames. Cosmic rays only appear in one frame so won't affect the median
        residual = frames[i] - neighbour_median #difference between current frame and reference
        #cosmic ray would produce a large residual
        mad = np.median(np.abs(residual)) + 1e-12 #Estimate the typical noise level using the Median Absolute Deviation (MAD)
        z_score = residual / mad #convert to a z score in units of MAD; cosmic rays give |z| >> 1

        n_spikes = np.sum(np.abs(z_score) > sigma_threshold) #count how many wavelength pixels exceed the detection threshold

        if n_spikes >= min_outliers: #flag if enough high-significance outliers are present
            cosmic_frames.append(i)

    print(f"Detected {len(cosmic_frames)} cosmic ray frames: {cosmic_frames}")
    return cosmic_frames


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

    return frames_clean


def background_subtract(frames: np.ndarray, bg_slice: tuple[int, int]) -> np.ndarray:
    '''
    Subtract background from each frame using the mean of a specified wavelength slice.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).
        bg_slice (tuple): Tuple specifying the start and stop indices for background calculation.

    Returns:
        np.ndarray: Background-subtracted frames.
    '''
    start, stop = bg_slice
    background = np.mean(frames[:, start:stop], axis=1)
    return frames - background[:, None]


def normalise(frames: np.ndarray) -> np.ndarray:
    '''
    Normalise each frame by its maximum value.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).

    Returns:
        np.ndarray: Normalised frames.
    '''
    max_vals = np.max(frames, axis=1)
    frames_norm = frames / (max_vals[:, None] + 1e-20)
    return frames_norm


def average_and_normalise(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Average all frames and normalise the result.
    Args:
        frames (np.ndarray): 2D array of spectral frames (num_frames x num_wavelengths).

    Result:
        tuple[np.ndarray, np.ndarray]: Averaged spectrum and normalised averaged spectrum.
    '''
    avg = np.mean(frames, axis=0)
    max_val = np.max(avg)
    avg_norm = avg / (max_val + 1e-20)
    return avg, avg_norm
