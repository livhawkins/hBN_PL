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
        
        fig = plt.figure(figsize=(8, 5))

        # Plot full spectrum
        plt.plot(self.x, self.y, label="PL Spectrum")

        # Mark peaks
        plt.scatter(self.x[self.peaks], self.y[self.peaks], label="Detected Peaks")

        plt.xlabel("Wavelength / Energy")
        plt.ylabel("Intensity (a.u.)")
        plt.title("PL Spectrum with Detected Peaks")

        plt.legend()
        plt.tight_layout()

        return fig

    def plot_zpl_on_spectrum(self, zpl_peaks):

        'Plot spectrum and highlight only ZPL peaks'

        if zpl_peaks is None or len(zpl_peaks) == 0:
            print("LOSER EMITTER NO ZPLS")
            return

        fig = plt.figure(figsize=(8,5))
        # full spectrum
        plt.plot(self.x, self.y, label="PL Spectrum")

        # convert ZPL locations → indices in x-array
        zpl_indices = [
            np.argmin(np.abs(self.x - p["location"]))
            for p in zpl_peaks
        ]

        # mark only ZPL peaks
        plt.scatter(
            self.x[zpl_indices],
            self.y[zpl_indices],
            label="ZPL Peaks",
            marker="x",
            s=120
        )

        # label them
        for i, idx in enumerate(zpl_indices, start=1):
            #$plt.text(self.x[idx + 33], self.y[idx], f"ZPL {i}")
            # shift label 1% of the total x-range to the right
            x_shift = (self.x[-1] - self.x[0]) * 0.01
            x_text = self.x[idx] + x_shift
            plt.text(x_text, self.y[idx], f"ZPL {i}")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity")
        plt.title("PL Spectrum with ZPL Peaks")
        plt.legend()
        plt.tight_layout()
        
        return fig

    def plot_energy(self, zpl_peaks):
        """
            Plot spectrum in energy units
        """
        if zpl_peaks is None or len(zpl_peaks) == 0:
            print("LOSER EMITTER NO ZPLS")
            return
        
        energy = 1239.8 / self.x

        fig = plt.figure(figsize=(8, 5))
        plt.plot(energy, self.y)

        zpl_indices = np.array([np.argmin(np.abs(self.x - p["location"])) for p in zpl_peaks])
        energy_zpl = energy[zpl_indices]  # use energy array, not 1239.8/self.x again



        #zpl_indices = [
        #    np.argmin(np.abs(self.x - p["location"]))
        #    for p in zpl_peaks
       #]

        energy_zpl = 1239.8 / self.x[zpl_indices]
        # mark only ZPL peaks
        plt.scatter(
            energy_zpl,
            self.y[zpl_indices],
            label="ZPL Peaks",
            marker="x",
            s=120
        )
        for i, idx in enumerate(zpl_indices, start=1):
            plt.text(energy[idx], self.y[idx], f"ZPL {i}")


        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title("PL Spectrum in Energy Units")
        plt.tight_layout()
        return fig
