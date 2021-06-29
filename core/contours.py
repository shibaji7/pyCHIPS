"""contours.py: Module is used to implement edge detection tecqniues using CV2 and apply Kernel estimations on the regions"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime as dt
import pandas as pd
import json
from scipy.stats import beta
plt.style.context("seaborn")
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"

import matplotlib.pyplot as plt
import cv2
import os

class ProbabilisticContours(object):
    """ Probabilistic Contours detection by Open-CV """
    
    def __init__(self, aia):
        np.random.seed(0)
        self.aia = aia
        self.filename = aia.fname
        self.folder = aia.folder
        self.file = aia.folder + aia.fname
        print(" Proc file - ", self.file)
        self.extn = "." + aia.fname.split(".")[-1]
        self.binfile = self.folder + self.filename.replace(self.extn, "_binmaps" + self.extn)
        if os.path.exists(self.binfile): 
            self.binmaps = cv2.imread(cv2.samples.findFile(self.binfile), cv2.IMREAD_GRAYSCALE)
            self.fcontours = (self.binmaps > 0).astype(int)
            if os.path.exists(self.aia.folder + self.aia.fname): self.aia.normalized()
            self.norm_data = self.aia.m_normalized.data * self.fcontours
            self.fcontours_nan = np.copy(self.fcontours).astype(float)
            self.fcontours_nan[self.fcontours_nan == 0] = np.nan
        else: raise Exception("File does not exists:", self.binfile)
        return
    
    def probability_contours(self, v_range=(10, 300), v_step=10, pth=0.5, summary=True):
        _dict_ = {}
        for thd in range(v_range[0], v_range[1], v_step):
            _data = self.norm_data * self.fcontours_nan
            _intensity = (thd - _data).ravel()
            _intensity = _intensity[~np.isnan(_intensity)]
            _probs =  1./(1.+np.exp(_intensity))
            _hist, _bin_edges = np.histogram(_probs, bins=np.linspace(0,1,21), density=True)
            _idx = np.diff(_bin_edges)
            _f = pd.DataFrame(); _f["pr"], _f["edgs"], _f["idx"] = _hist, _bin_edges[:-1], _idx
            _f = _f[_f.edgs >= pth]
            _mask = (self.norm_data <= thd).astype(int) * self.fcontours
            _dict_[thd] = (np.sum(_f.idx*_f.pr), _mask)
        if summary: self.generate_analysis_summary("norm", _dict_)
        return _dict_
    
    def probability_contours_with_intensity_operation(self, v_range=(10, 300), v_step=10, pth=0.5,                                                       operations={"name":r"Logarithm: $10\times$ $log_{10}(x)$","fn":lambda x: 10*np.log10(x)}):
        _dict_ = {}
        
        return
    
    def generate_analysis_summary(self, kind="norm", _dict_ = {}):
        """
        Plot histograms for normalized intensity, thresholds and other parameters
        """
        summary_folder = self.folder + "summary/"
        if not os.path.exists(summary_folder): os.mkdir(summary_folder)
        if kind=="norm": norm_data = np.copy(self.norm_data)
        for key in _dict_.keys():
            _p, _mask = _dict_[key]
            fig, ax = plt.subplots(dpi=180, figsize=(4,4), nrows=1, ncols=1)
            ax.imshow(_mask*255, extent=[-1024,1024,-1024,1024], cmap="gray")
            ax.set_xticks([-1024,-512,0,512,1024])
            ax.set_yticks([-1024,-512,0,512,1024])
            ax.set_xticklabels([r"-$2^{10}$",r"-$2^{9}$","0",r"$2^{9}$",r"$2^{10}$"])
            ax.set_yticklabels([r"-$2^{10}$",r"-$2^{9}$","0",r"$2^{9}$",r"$2^{10}$"])
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.set_title(r"$\mathcal{F}(x_{\tau},I_{th})$=%.3f, $x_{\tau}$=0.5, $I_{th}$=%d"%(_p,key), fontdict={"size":8})
            fig.savefig(summary_folder + self.filename.replace(self.extn, "_binmaps_%04d"%key + self.extn), bbox_inches="tight")
            plt.close()
        os.system("zip -r summary.zip " + summary_folder)
        return