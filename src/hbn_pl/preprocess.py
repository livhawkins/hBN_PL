import numpy as np
import spectrapepper as spep


def remove_frames(frames, frames_to_remove):
    # Doc strings plz
    if frames_to_remove is None or len(frames_to_remove) == 0:
        return frames

    return np.delete(frames, frames_to_remove, axis=0)


def remove_cosmic(frames, cosmic_frames, sigma=1.7):
    # Doc strings plz
    if cosmic_frames is None or len(cosmic_frames) == 0:
        return frames

    frames_clean = frames.copy()
    n_frames = frames.shape[0]

    for i in cosmic_frames:
        if i <= 0 or i >= n_frames - 1:
            # Cannot apply median method at boundaries
            continue

        spectra_list = [
            frames[i - 1],
            frames[i],
            frames[i + 1],
        ]

        corrected = spep.cosmicmed(spectra_list, sigma=sigma)
        frames_clean[i] = corrected[1]

    return frames_clean


def background_subtract(frames, bg_slice=(10, 200)):
    # Doc strings plz
    start, stop = bg_slice
    background = np.mean(frames[:, start:stop], axis=1)
    return frames - background[:, None]


def average_and_normalise(frames):
    # Doc strings plz
    avg = np.mean(frames, axis=0)

    max_val = np.max(avg)

    avg_norm = avg / (max_val + 1e-20)

    return avg, avg_norm
