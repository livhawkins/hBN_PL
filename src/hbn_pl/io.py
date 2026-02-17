import numpy as np
from pathlib import Path
from hbn_pl.SPE3reading import SPE3map
import matplotlib.pyplot as plt
import csv

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



#def output(input: type) -> what is the output form
#to run in notebook: output(p, OUTPUT_DIR)
def output(peakdata: dict, 
           classification_msg: str, 
           output_dir: str, 
           input_file: str, 
           wavelength=None,
           frames=None, 
           plot=None,
           peaks2= None,
           finder=None,
           zpl_peaks=None):
    ''' Saves Peakfinder output, including:
    - CSV with peak data + classification
    - Directory structure:
        emitter_name/
            peak_data.csv
            original_frames/
            found_peaks/
            zpl_plots/ (wavelength and energy)

    Args:
        peakdata (dict): Dictionary containing peak information and ZPL classification.
        classification_msg (str): The classification message for the peaks.
        output_dir (str or Path): Base output directory path.
        input_file (str or Path): The input file path for the emitter.
        wavelength (np.ndarray, optional): Wavelength data for plotting original frames.
        frames (np.ndarray, optional): Frame data for plotting original frames.
        plot (function, optional): Function to plot original frames.
        peaks2 (list, optional): List of peak locations to be plotted.
        finder (object, optional): PeakFinder object with plotting methods.
        zpl_peaks (list, optional): List of ZPL peak locations.

    Returns:
        Print statements: Saves outputs to specified path within emitter output directory, with internal directory organisation.
    '''

    # first check if output directory exists
    messages = []  # collect status messages

    # Check/create base output directory
    base_output_directory = Path(output_dir)
    if not base_output_directory.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_output_directory}")

    # Create emitter directory
    emitter_name = Path(input_file).stem
    emitter_outputfolder = base_output_directory / emitter_name
    emitter_outputfolder.mkdir(parents=True, exist_ok=True)
    messages.append(f"Created emitter output directory: {emitter_outputfolder}")

    # Save CSV
    csv_file = emitter_outputfolder / f"{emitter_name}_peak_data.csv"
    if isinstance(peakdata, dict):
        peakdata = [peakdata]

    with open(csv_file, mode='w', newline='') as f:
        csv_write = csv.writer(f)
        # metadata section? ZPL classification prints as a single line, before the CSV table (as comment-style metadata for CSV format)
        csv_write.writerow([f"#ZPL Classification: {classification_msg}"])
        csv_write.writerow([]) # blank line before table

        #table header:
        csv_write.writerow(["Peak Location", "Peak Intensity", "Peak Prominence"])
        for peak in peakdata:
            csv_write.writerow([
                peak["location"],
                peak["intensity"],
                peak["prominence"],
            ])
    messages.append(f"Saved CSV file in emitter output directory")

    # Create plot subdirectories
    original_dir = emitter_outputfolder / "original_frames"
    found_dir = emitter_outputfolder / "found_peaks"
    ZPL_dir = emitter_outputfolder / "ZPL_plots"

    original_dir.mkdir(exist_ok=True)
    found_dir.mkdir(exist_ok=True)
    ZPL_dir.mkdir(exist_ok=True)

    # 1: Original frames
    if wavelength is None or frames is None or plot is None:
        messages.append("Original frames not saved: wavelength, frames, or plot function missing.")
    else:
        figs = plot.plot_frames(wavelength, frames, show=False)
        for i, fig in enumerate(figs):
            fig.savefig(original_dir / f"{emitter_name}_frame_{i}.png")
        plt.close('all')
        messages.append(f"Saved original frame plots ({len(figs)} frames) in: emitter output directory/original_frames")

    # 2: Spectra with detected peaks
    if finder is None:
        messages.append("Finder object not provided, peaks plots skipped.")
    elif finder.peaks is None or len(finder.peaks) == 0:
        messages.append("No peaks found, peaks plot skipped.")
    else:
        peaks_fig = finder.plot_peaks()
        peaks_fig.savefig(found_dir / f"{emitter_name}_found_peaks.png")
        plt.close(peaks_fig)
        messages.append(f"Saved peak plot in: emitter output directory/found_peaks")

    # 3: ZPL wavelength and energy plots
    if finder is None or zpl_peaks is None or len(zpl_peaks) == 0:
        messages.append("No ZPL peaks found or finder not provided, ZPL plots skipped.")
    else:
        # ZPL wavelength
        zpl_fig = finder.plot_zpl_on_spectrum(zpl_peaks)
        if zpl_fig is not None:
            zpl_fig.savefig(ZPL_dir / f"{emitter_name}_ZPL_wavelength_plot.png")
            plt.close(zpl_fig)
            messages.append(f"Saved ZPL wavelength plot in: emitter output directory/ZPL_plots")

        # Energy plot
        energy_fig = finder.plot_energy(zpl_peaks)
        if energy_fig is not None:
            energy_fig.savefig(ZPL_dir / f"{emitter_name}_ZPL_energy_plot.png")
            plt.close(energy_fig)
            messages.append(f"Saved ZPL energy plot in: emitter output directory/ZPL_plots")

    # Final summary
    summary = "\n".join(messages)
    print(f"Peakfinder outputs for emitter '{input_file}' saved in:\n{emitter_outputfolder}. \n\n{summary}")
    


