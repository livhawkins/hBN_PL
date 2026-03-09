from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import clear_output
import hbn_pl.plot as plot
import hbn_pl.peaks as peaks

def show_parameters() -> None:
    '''
    Display current fit parameters.
    This is called for every spectrum to remind the user of the current settings before they choose to edit or refit.
    '''
    print("\nCurrent Fit Parameters")
    print("----------------------")
    print("targets =", targets)
    print("window  =", window)
    print("sigma   =", sigma)

def progress_bar(i: int, total: int) -> None:
    '''
    Display a text progress bar in the console.

    Args:
        i: current index (0-based)
        total: total number of items

    Returns:
        None
    '''
    bar_length = 30
    progress = (i + 1) / total

    filled = int(bar_length * progress)

    bar = "█" * filled + "-" * (bar_length - filled)

    percent = int(progress * 100)

    print(f"\nProgress |{bar}| {percent}% ({i+1}/{total}) files\n")


def fit_all_peaks(energy: np.ndarray, intensity: np.ndarray) -> list:
    '''
    Fit Gaussian peaks to the spectrum at the specified target positions.

    Args:   
        energy: 1D array of energy values
        intensity: 1D array of intensity values
    
    Returns:
        List of fit results for each target peak. Each result is a dictionary containing:
        - 'center': fitted peak center in meV
        - 'center_err': uncertainty in the fitted peak center
        - 'amplitude': fitted peak amplitude
        - 'amplitude_err': uncertainty in the fitted peak amplitude
        - 'sigma': fitted peak width (sigma)
        - 'sigma_err': uncertainty in the fitted peak width
    '''
    results = []

    for t in targets:

        res = peaks.fit_peak_gaussian(
            energy,
            intensity,
            center_guess=t,
            window=window,
            sigma_guess=sigma
        )

        results.append(res)

    return results

def select_peaks(results: list) -> list:
    '''
    Allow the user to select which peaks to keep from the fitted results.

    Args:
        results: list of fit results for each target peak (output from fit_all_peaks)
    
    Returns:
        List of indices for the peaks to keep.
    '''
    print("\nDetected peaks:\n")

    for i, r in enumerate(results):

        print(f"{i}: {r['center']:.2f} ± {r['center_err']:.2f} meV")

    text = input(
        "\nEnter indices to KEEP (comma separated, e.g. 0,1,3): "
    )

    try:
        keep = [int(x.strip()) for x in text.split(",")]
    except:
        print("Invalid input. Keeping all peaks.")
        keep = list(range(len(results)))

    return keep


def edit_parameters() -> None:
    '''
    Allow the user to edit the fit parameters (targets, window, sigma) during the batch process.

    Args:
        None (uses global variables)

    Returns:
        None (updates global variables)
    '''
    global targets, window, sigma

    print("\nCurrent parameters")

    print("targets =", targets)
    print("window =", window)
    print("sigma =", sigma)

    t = input("\nNew targets (comma list) or ENTER: ")
    w = input("New window or ENTER: ")
    s = input("New sigma or ENTER: ")

    if t != "":
        targets = [float(x) for x in t.split(",")]

    if w != "":
        window = float(w)

    if s != "":
        sigma = float(s)

def save_peak_csv(directory: Path, file_name: str, zpl: float, results: list, keep_indices: list) -> None:
    '''
    Save the fitted peak results to a CSV file. Each row corresponds to one spectrum, 
    with columns for the file name, ZPL wavelength, and the fitted peak parameters for each target peak (center energy and error).

    Args:
        directory: Path to the directory where the CSV file will be saved
        file_name: Name of the spectrum file (used for reference in the CSV)    
        zpl: ZPL wavelength for the spectrum (used for reference in the CSV)
        results: List of fit results for each target peak (output from fit_all_peaks)
        keep_indices: List of indices for the peaks that were selected to keep (output from select_peaks)

    Returns:
        None (saves/updates the CSV file in the specified directory)
    '''
    csv_path = directory / "peak_fitting_results.csv"

    row = {
        "file": file_name,
        "zpl_wavelength": zpl
    }

    n_peaks = len(targets)

    for i in range(n_peaks):

        if i in keep_indices:

            r = results[i]

            row[f"peak{i+1}_meV"] = r["center"]
            row[f"peak{i+1}_err"] = r["center_err"]

        else:

            row[f"peak{i+1}_meV"] = np.nan
            row[f"peak{i+1}_err"] = np.nan

    df_new = pd.DataFrame([row])

    if csv_path.exists():

        df_old = pd.read_csv(csv_path)

        # ensure columns stay consistent if targets change
        for col in df_new.columns:
            if col not in df_old.columns:
                df_old[col] = np.nan

        for col in df_old.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan

        df = pd.concat([df_old, df_new], ignore_index=True)

    else:

        df = df_new

    df.to_csv(csv_path, index=False)


def process_file(file_path: Path, i: int, total: int) -> tuple:
    '''
    Process a single spectrum file: load data, fit peaks, display interactive plot, 
    and handle user commands for saving/skipping/editing.
    
    Args:
        file_path: Path to the spectrum file to process
        i: current index of the file in the batch (for progress display)   
        total: total number of files in the batch (for progress display)

    Returns:
        A tuple containing:
        - action: string indicating the user's choice ("save", "skip", "back", "quit")
        - results: list of fit results for the peaks (if action is "save", otherwise None)
        - keep_indices: list of indices for the peaks to keep (if action is "save", otherwise None)
        - zpl: ZPL wavelength for the spectrum (if action is "save", otherwise None)
    '''
    data = np.load(file_path)

    energy = data["energy"]
    intensity = data["intensity"]
    zpl = data["zpl_wavelength"]

    print("\nProcessing:", file_path.name)
    print("ZPL:", zpl)

    keep_indices = list(range(len(targets)))

    while True:

        results = fit_all_peaks(energy, intensity)

        base = file_path.stem

        html_file = file_path.parent / f"{base}_interactive.html"

        plot.plot_psb_plotly(
            energy,
            intensity,
            results,
            targets,
            window=window,
            filename=html_file
        )

        show_parameters()
        progress_bar(i, total)

        print("Commands")
        print("ENTER = save")
        print("s = skip file")
        print("p = choose peaks")
        print("e = edit parameters")
        print("r = refit")
        print("b = back")
        print("q = quit")

        cmd = input("\nChoice: ").strip().lower()

        if cmd == "":
            return "save", results, keep_indices, zpl

        if cmd == "s":
            return "skip", None, None, None

        if cmd == "p":

            keep_indices = select_peaks(results)

            return "save", results, keep_indices, zpl

        if cmd == "e":

            edit_parameters()
            print("\nParameters updated. Refitting...\n")
            continue

        if cmd == "r":

            continue

        if cmd == "b":
            return "back", None, None, None

        if cmd == "q":
            return "quit", None, None, None


def batch_peak_fitting(directory: Path) -> None:
    '''
    Process a batch of spectrum files for peak fitting.

    Args:
        directory: Path to the directory containing the spectrum files (expects files named "*_cleaned.npz")
    
    Returns:
        None (saves peak fitting results to a CSV file in the same directory)
    '''
    files = sorted(directory.glob("*_cleaned.npz"))

    total = len(files)

    print("Found", total, "files")

    i = 0

    while i < total:

        clear_output(wait=True)

        file = files[i]

        action, results, keep_indices, zpl = process_file(file, i, total)

        if action == "quit":
            break

        if action == "back":

            i = max(0, i - 1)
            continue

        if action == "skip":

            i += 1
            continue

        if action == "save":

            save_peak_csv(directory, file.name, zpl, results, keep_indices)

            print("\nSaved peak results")

            i += 1

    print("\nBatch finished.")