import sys
sys.path.append(r"C://Users/ech77/OneDrive - University of Cambridge/PhD work/Code/hBN_PL/src")

from pathlib import Path
import numpy as np
import hbn_pl.plot as plot
import hbn_pl.peaks as peaks

DATA_DIR = Path("C://Users/ech77/OneDrive - University of Cambridge/PhD work/Data/Data for IBS November 2025/Sample 5/PL spectra")
FILE_NAME = '2025-11-18 18_43_24 sample5_DCMd2_200uw_emitter37 2216_cleaned_averaged.npz'

spe_path = DATA_DIR / FILE_NAME
data = np.load(spe_path)
print(data.keys())
wavelength = data["wavelength"]
intensity = data["spectra"]
intensity = intensity / np.max(intensity)  # Normalize intensity for better visualization

plot.plot_spectrum(wavelength, intensity)

mask = (wavelength >= 615) & (wavelength <= 650)
filtered_wavelength = wavelength[mask]
filtered_intensity = intensity[mask]
peak_idx = np.argmax(filtered_intensity)
zpl_wavelength = filtered_wavelength[peak_idx]
print(f"ZPL wavelength: {zpl_wavelength:.2f} nm")

intensity, energy, figure = plot.plot_energy(wavelength, intensity, zpl_wavelength)

targets = [12, 41.5, 150, 172]
window = 3
sigma = 2

fit_results = []

for t in targets:
    res = peaks.fit_peak_gaussian(energy, intensity, center_guess=t, window=window, sigma_guess=sigma)
    print(f"Fitted center for target {t} meV: {res['center']:.2f} ± {res['center_err']:.2f} meV")
    fit_results.append(res)

spe_path = Path(spe_path)
base_name = spe_path.stem  # strip directory & extension
html_filename = spe_path.parent / f"{base_name}_interactive.html"
plot.plot_psb_plotly(energy, intensity, fit_results, targets, window=window, filename=html_filename) #Save interactive HTML