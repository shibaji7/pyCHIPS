"""chips_utils.py: Module is used to implement utility functions"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])

import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def download_riess21(fname="tmp/Reiss2021/riess21.zip",
                     url="https://figshare.com/ndownloader/articles/13397261/versions/2"):
    """
    Download all data from reomte to compare
    """
    if not os.path.exists(fname): 
        os.system(f"wget -O {fname} {url}")
        shutil.unpack_archive(fname, "/".join(fname.split("/")[:-1]))
    return

def get_data_by_model(model="ASSA", plot=False):
    """
    Get 2D array data by model name
    """
    o = np.empty((4096, 4096))
    fname = f"tmp/Reiss2021/binary{model}.txt"
    download_riess21()
    if os.path.exists(fname):
        o = np.loadtxt(fname)
        print(f" Load file - {fname}, [{o.shape}]")
        if plot: plot_saved_data("tmp/2018-05-30-12-00/", o, model)
    else: print(f" File - {fname}, does not exists.")
    return o

def plot_saved_data(folder, o, model):
    """
    Plot modeled data from Riess 2021
    """
    file = folder + f"{model}_analysis.png"
    k = 0
    fig, ax = plt.subplots(dpi=120, figsize=(3,3), nrows=1, ncols=1, sharex="all", sharey="all")
    cmap = matplotlib.cm.gray
    cmap.set_bad("k",1.)
    ax.imshow(o, origin="lower", cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.savefig(file, bbox_inches="tight")
    return

def plot_compare_data(folder, o, c, model, lim, sim=None, th=None):
    """
    Plot modeled data from Riess 2021 and CHIPS
    """
    file = folder + f"{model}_chips_{lim}_comp.png"
    fig, axes = plt.subplots(dpi=150, figsize=(6,3), nrows=1, ncols=2, sharex="all", sharey="all")
    cmap = matplotlib.cm.gray
    cmap.set_bad("k",1.)
    ax = axes[0]
    ax.imshow(o, origin="lower", cmap=cmap)
    ax.text(0.01, 1.05, "CHIPS", ha="left", va="center", transform=ax.transAxes)
    if th is not None: ax.text(0.99, 1.05, r"%s"%th, ha="right", va="center", transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xticks([])
    ax = axes[1]
    ax.imshow(c, origin="lower", cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(0.01, 1.05, model, ha="left", va="center", transform=ax.transAxes)
    if sim is not None: ax.text(0.99, 1.05, r"%s"%sim, ha="right", va="center", transform=ax.transAxes)
    fig.savefig(file, bbox_inches="tight")
    return

def measure_similaity(im_x, im_y):
    """
    Measure similarity between images x and y
    """
    sim = {}
    sim["cos"] = cosine_similarity(im_x.reshape(1, -1), im_y.reshape(1, -1))[0,0]
    return sim
    
if __name__ == "__main__":
    get_data_by_model()