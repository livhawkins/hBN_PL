import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import pickle

import hbn_pl.io as io
import hbn_pl.preprocess as preprocess
import hbn_pl.plot as plot


good_emitters = []
review_log = []


def review_prompt(current_index: int, total: int) -> tuple[str, bool]:
    '''
    Prompt the user to review the current emitter and choose an action.
    
    Args:
        current_index: The index of the current emitter being reviewed (0-based)
        total: The total number of emitters in the batch

    Returns:
        A tuple containing:
        - action: A string indicating the user's choice ("save", "skip", "back", "quit")
        - good_flag: A boolean indicating whether the emitter was marked as good (True if "g" was chosen, False otherwise)
    '''
    bar_length = 30
    progress = (current_index + 1) / total
    filled_length = int(bar_length * progress)

    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    percent = int(progress * 100)

    print(f"Progress: |{bar}| {percent}% ({current_index + 1}/{total} emitters)\n")

    print("Options:")
    print("ENTER → Save")
    print("s     → Skip")
    print("g     → Mark good emitter + save")
    print("b     → Go back to previous emitter")
    print("q     → Quit batch")

    user = input("Choice: ").strip().lower()

    if user == "":
        return "save", False
    elif user == "s":
        return "skip", False
    elif user == "g":
        return "save", True
    elif user == "b":
        return "back", False
    elif user == "q":
        return "quit", False
    else:
        return "save", False


def save_csvs(directory: Path) -> None:
    '''
    Save the review log and good emitters list to CSV files in the specified directory.

    Args:
        directory: Path to the directory where the CSV files will be saved

    Returns:
        None (saves CSV files to disk)
    '''
    if review_log:
        df_log = pd.DataFrame(review_log)
        df_log.to_csv(directory / "review_log.csv", index=False)

    if good_emitters:
        df_good = pd.DataFrame({"good_emitters": good_emitters})
        df_good.to_csv(directory / "good_emitters.csv", index=False)


def process_pickle_emitter(data: dict, emitter_key: str, emitter_number: int,
                           wait_for_confirm: bool=True,
                           current_index: int=0,
                           total: int=0) -> tuple[str, bool, float, np.ndarray, np.ndarray, plt.Figure]:
    '''
    Process a single emitter from the pickle data.

    Args:
        data: The dictionary containing the pickle data
        emitter_key: The key for the emitter in the data dictionary
        emitter_number: The number of the emitter being processed
        wait_for_confirm: If True, prompts the user to review the results and choose an action
        current_index: The index of the current emitter being processed (for progress display)
        total: The total number of emitters in the batch (for progress display)

    Returns:
        A tuple containing:
        - action: A string indicating the user's choice ("save", "skip", "back", "quit")
        - good_flag: A boolean indicating whether the emitter was marked as good (True if "g" was chosen, False otherwise)
        - zpl_wavelength: The estimated ZPL wavelength in nm
        - intensity: The intensity values of the spectrum
        - energy: The energy values corresponding to the spectrum
        - figure: The matplotlib figure object containing the energy plot
    '''
    global good_emitters, review_log

    print(f"\nProcessing: Emitter {emitter_number}")

    wavelength = np.array(data["Wavelength"])

    frame_keys = sorted(data[emitter_key].keys())
    frames = np.array([data[emitter_key][k] for k in frame_keys])

    # Background subtraction
    frames = preprocess.background_subtract(frames, bg_slice=(1, 50))

    # Cosmic ray detection
    cosmic_frames, cosmic_location = preprocess.detect_cosmic_frames(
        frames,
        wavelength,
        prominence_threshold=0.05,
        fwhm_threshold=1.9,
        n_peaks=3,
        z_thresh=100,
        half_width=5,
        noise_width=30
    )

    plot.plot_cosmic_frames(frames, wavelength, cosmic_location)

    frames_cleaned, cosmic_figs = preprocess.remove_cosmic_rays(
        frames,
        wavelength,
        cosmic_frames,
        cosmic_location,
        sigma=1.5,
        half_width=5
    )

    frames = frames_cleaned

    # Remove bad frames
    drop_fraction = 0.6

    bad_frames = preprocess.detect_bad_frames_simple(
        frames,
        drop_fraction=drop_fraction
    )

    frames = preprocess.remove_frames(frames, bad_frames)

    # Align ZPLs for many frames to fix spectral jumping

    aligned_frames, zpls, ref_zpl = preprocess.align_zpl(frames, wavelength)

    # Average and normalize
    avg, avg_norm = preprocess.average_and_normalise(aligned_frames)

    plot.plot_spectrum(
        x=wavelength,
        spectrum=avg_norm,
        x_quantity='Wavelength (nm)'
    )

    # Estimate ZPL
    max_index = np.argmax(avg_norm)
    zpl_wavelength = wavelength[max_index]

    print("Estimated ZPL:", zpl_wavelength)

    # Energy plot
    intensity, energy, figure = plot.plot_energy(
        wavelength,
        avg_norm,
        zpl_wavelength
    )

    figure.canvas.draw()
    plt.show(block=False)
    plt.pause(0.1)

    if wait_for_confirm:

        action, good_flag = review_prompt(current_index, total)

        return action, good_flag, zpl_wavelength, intensity, energy, figure

    else:

        return "save", False, zpl_wavelength, intensity, energy, figure


def batch_process_pickle(pickle_path: Path, wait_for_confirm: bool=True) -> None:
    '''
    Process a batch of spectra from a pickle file in the specified directory, allowing the user to review each emitter and choose actions. 
    Saves results and logs to CSV files.

    Args:
        pickle_path: Path to the pickle file containing the spectra data
        wait_for_confirm: If True, prompts the user to review each emitter and choose actions. If False, processes all files and saves results without prompting.
    
    Returns:
        None (saves results and logs to CSV files in the specified directory)
    '''
    global good_emitters, review_log

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    emitter_keys = sorted(k for k in data.keys() if k != "Wavelength")

    total_emitters = len(emitter_keys)

    print(f"Found {total_emitters} emitters")

    i = 0

    while i < total_emitters:

        clear_output(wait=True)
        plt.close('all')

        emitter_key = emitter_keys[i]
        emitter_name = f"Emitter {i+1}"

        action, good_flag, zpl_wavelength, intensity, energy, figure = \
            process_pickle_emitter(
                data,
                emitter_key,
                i + 1,
                wait_for_confirm=wait_for_confirm,
                current_index=i,
                total=total_emitters
            )

        if action == "quit":
            print("User aborted batch processing.")
            break

        if action == "back":

            if i > 0:
                i -= 1

                if review_log:
                    last_entry = review_log.pop()

                    if last_entry["good_emitter"] and last_entry["emitter"] in good_emitters:
                        good_emitters.remove(last_entry["emitter"])

                continue

            else:
                print("Already at the first emitter!")
                continue

        review_log.append({
            "emitter": emitter_name,
            "action": action,
            "good_emitter": good_flag,
            "zpl_nm": zpl_wavelength
        })

        if good_flag:
            good_emitters.append(emitter_name)
            print("⭐ Marked as good emitter")

        save_csvs(pickle_path.parent)

        if action != "skip":
            emitter_path = pickle_path.parent / f"Emitter_{i+1}.spe"
            io.save_preprocess_results(
                emitter_path,
                intensity,
                energy,
                figure,
                zpl_wavelength)

            print(f"\nSaved results for {emitter_name}")

        else:

            print(f"\nSkipped saving for {emitter_name}")

        i += 1

    print("\nBatch processing complete.")