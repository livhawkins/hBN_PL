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
            msg = "Bad Data (no ZPLs detected)"
        elif n_zpl == 1:
            msg = "Single ZPL"
        elif n_zpl == 2:
            msg = "Two ZPLs"
        elif n_zpl == 3:
            msg = "Three ZPLs"
        elif n_zpl > 3:
            msg = "Bad Data (too many ZPLs detected)"
        
        # pretty peak data hihi
        if n_zpl > 0:
            peak_lines = []

            for i, p in enumerate(self.zpl_peaks, start=1):

                peak_lines.append(f"Peak {i}: location = {p['location']:.1f}, prominence = {p['prominence']:.1f}")
                
            peak_text = "\n".join(peak_lines)
            msg = f"{msg}\n{peak_text}"
        
        return msg