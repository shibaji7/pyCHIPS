"""
    chips.py: Module is used to implement edge detection tecqniues using thresholding.
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import chips.utils as utils


class Chips(object):
    """
    Analyze the Maps
    """

    def __init__(
        self,
        aia,
        clear=False,
    ):
        self.aia = aia
        self.folder = utils.get_folder(self.aia.date)
        self.clear = clear
        if self.clear:
            self.clear_last_run()
        os.makedirs(self.folder, exist_ok=True)
        return

    def load(self):
        """
        Load latest run to sol_properties
        """
        fname = self.folder + ".nc"
        if os.path.exists(fname):
            with open(fname, "rb") as h:
                self.sol_properties = pickle.load(h)
        else:
            self.save_current_run()
        return

    def clear_last_run(self):
        """
        Clear latest run
        """
        import shutil
        shutil.rmtree(self.folder)
        return
    
    def processes_CHIPS(
        self, 
        wv=193, 
        res=4096,
    ):
        """
        Process CHIPS code
        """
        self.wavelength = wv
        self.resolution = res
        data = self.data[wv][res]
        self.extract_basic_properties()
        self.extract_solar_masks()
        self.run_filters()
        self.extract_histogram()
        self.extract_histograms()
        self.extract_sliding_histograms()
        self.extract_CHs_CHBs()
        return
