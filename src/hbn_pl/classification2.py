class ZPLClassification:
    """
    Classifies number of Zero-Phonon Lines (ZPLs) in a spectrum.

    Args:
    peaks : list of dict
        Each peak must contain:
        {
            "location": float,
            "fwhm": float,
            "prominence": float
        }

    thresholds : dict
        Threshold values used for classification.
    """

    def __init__(self, peaks, thresholds):

        self.peaks = peaks

        # Not sure if this are good parameters
        self.thresholds = thresholds

        self.zpl_peaks = []

    
    def _is_zpl(self, peak):
        """Check if a peak satisfies ZPL conditions"""

        if peak["fwhm"] > self.thresholds["max_fwhm"]:
            return False

        if peak["prominence"] < self.thresholds["min_prominence"]:
            return False

        loc_range = self.thresholds["location_range"]
        if loc_range is not None:
            if not (loc_range[0] <= peak["location"] <= loc_range[1]):
                return False

        return True

    
    def classify(self):
        """Return number of detected ZPLs"""

        self.zpl_peaks = [p for p in self.peaks if self._is_zpl(p)]

        n_zpl = len(self.zpl_peaks)

        if n_zpl == 0:
            return "Bad Data (no ZPLs detected)"
        elif n_zpl == 1:
            return "Single ZPL", print(self.zpl_peaks)
        elif n_zpl == 2:
            return "Two ZPLs", print(self.zpl_peaks)
        elif n_zpl == 3:
            return "Three ZPLs", print(self.zpl_peaks)
        elif n_zpl > 3:
            return "Bad Data (too many ZPLs detected)", print(self.zpl_peaks)