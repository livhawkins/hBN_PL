from scipy.signal import find_peaks, peak_widths
import numpy as np
import matplotlib.pyplot as plt

class PeakFinder:
    """
    Finds and characterises peaks in PL spectra.
    """

    def __init__(self, x_data, y_data):
        """
        Args:
        x_data : array
            Wavelength or energy axis
        y_data : array
            PL intensity
        """
        self.x = np.array(x_data)
        self.y = np.array(y_data)

        self.peaks = None
        self.properties = None
        self.peak_list = []

   
    def peak_finding(self, prominence=0.05, height=None, distance=None):
        """
        Run scipy peak finding
        """

        self.peaks, self.properties = find_peaks(
            self.y,
            prominence=prominence,
            height=height,
            distance=distance
        )

        return self.peaks

    
    def extract_peak_parameters(self):
        """
        Calculate FWHM and output structured peak info
        """

        if self.peaks is None:
            raise ValueError("Bad data")

        # FWHM calculation
        widths, width_heights, left_ips, right_ips = peak_widths(
            self.y,
            self.peaks,
            rel_height=0.5
        )

        self.peak_list = []

        for i, peak_index in enumerate(self.peaks):

            fwhm = (
                self.x[int(right_ips[i])] -
                self.x[int(left_ips[i])]
            )

            peak_dict = {
                "location": self.x[peak_index],
                "intensity": self.y[peak_index],
                "fwhm": fwhm,
                "prominence": self.properties["prominences"][i]
            }

            self.peak_list.append(peak_dict)

        return self.peak_list
    
    
    def plot_peaks(self):
        """
            Plot spectrum and highlight detected peaks
        """

        if self.peaks is None:
            raise ValueError("Error: No peaks found")
        
        plt.figure(figsize=(8, 5))

        # Plot full spectrum
        plt.plot(self.x, self.y, label="PL Spectrum")

        # Mark peaks
        plt.scatter(self.x[self.peaks], self.y[self.peaks], label="Detected Peaks")

        plt.xlabel("Wavelength / Energy")
        plt.ylabel("Intensity (a.u.)")
        plt.title("PL Spectrum with Detected Peaks")

        plt.legend()
        plt.tight_layout()
        plt.show()
