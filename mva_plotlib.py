"""plotlib.py: Module is used to implement different images for the MVA-Japan Conference paper"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
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
plt.style.context("seaborn")
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"

from to_remote import get_session
from data_pipeline import fetch_filenames
import sys
sys.path.append("core/")
from cv2_edge_detectors import EdgeDetection

def set_axes(ax, image, title, gray_map=False):
    ax.set_ylabel("Y (arcsec)", fontdict={"size":10})
    ax.set_xlabel("X (arcsec)", fontdict={"size":10})
    ax.set_title(title, fontdict={"size":10})
    if gray_map: ax.imshow(image, extent=[-1024,1024,-1024,1024], cmap="gray")
    else: ax.imshow(image, extent=[-1024,1024,-1024,1024])
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xticks([-1024,-512,0,512,1024])
    ax.set_yticks([-1024,-512,0,512,1024])
    return

def introduction_image(dn):
    conn = get_session()
    wvbins = [131, 171, 193, 211, 335, 94]
    titles = ["a. AIA 131 (0.4 MK)", "b. AIA 171 (0.7 MK)", "c. AIA 193 (1.2 MK)", "d. AIA 211 (2.0 MK)", 
             "e. AIA 335 (2.5 MK)", "f. AIA 94 (6.3 MK)"]
    fig, axes = plt.subplots(dpi=180, figsize=(5, 8), nrows=3, ncols=2)
    for j, wv in enumerate(wvbins):
        files, folder = fetch_filenames(dn, 1024, wv)
        os.system("mkdir -p " + folder)
        dates = [dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S") for f in files]
        x = pd.DataFrame(np.array([dates, files]).T, columns=["date", "fname"])
        x["delt"] = np.abs([(u-dn).total_seconds() for u in x.date])
        i = x.delt.idxmin() 
        x = x.iloc[[i]]
        f = folder + x.fname.tolist()[0]
        if conn.chek_remote_file_exists(f): conn.from_remote_FS(f)
        image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        if wv in [193,211]: image = cv2.rectangle(image, (360,440), (712,712), (255,255,0), 2)
        rows = image.shape[0]
        image[rows-int(rows/24):rows-1,0:int(rows/2),:] = 0
        ax = axes[np.mod(j,3), int(j/3)]
        set_axes(ax, image, titles[j])
        os.remove(f)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.savefig("data/mva.Figure01.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    conn.close()
    return

def analysis_image(dn, wv=193):
    files, folder = fetch_filenames(dn, 1024, wv)
    dates = [dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S") for f in files]
    x = pd.DataFrame(np.array([dates, files]).T, columns=["date", "fname"])
    x["delt"] = np.abs([(u-dn).total_seconds() for u in x.date])
    i = x.delt.idxmin() 
    x = x.iloc[[i]]
    files = [x.fname.tolist()[0]]
    _dict_ = {"date":dn, "resolution": 1024, "wavelength":wv, "loc": "sdo", "verbose": True, "draw": True}
    ed = EdgeDetection(files, folder, _dict_).find()
    ed.close()
    titles = ["a. Original (AIA 193)", "b. Gray-Scale:HC", "c. Filtered ($K_{size}=%d$)"%ed.gauss_kernel,
              "d. Masked:HC", "e. Threshold: OTSU", "f. Conturs", 
              r"g. $\mathcal{F}\left(I^C> %d\right)\geq %.2f$"%(ed.intensity_threshold, ed.intensity_prob_threshold), "h. Detected CHB"]
    fig, axes = plt.subplots(dpi=180, figsize=(5, 10), nrows=4, ncols=2)
    set_axes(axes[0,0], cv2.rectangle(ed.org, (360,440), (712,712), (255,255,0), 2), titles[0])
    set_axes(axes[0,1], ed.gray_hc, titles[1], True)
    set_axes(axes[1,0], ed.blur, titles[2], True)
    set_axes(axes[1,1], ed.blur_mask, titles[3], True)
    set_axes(axes[2,0], ed.inv, titles[4], True)
    set_axes(axes[2,1], cv2.cvtColor(ed.src, cv2.COLOR_BGR2RGB), titles[5])
    set_axes(axes[3,0], ed.prob_masked_gray_image, titles[6], True)
    set_axes(axes[3,1], ed.prob_masked_image, titles[7])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.savefig("data/mva.Figure03.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

def final_image_parameters(dn, ax=None, th=32, wv=193):
    files, folder = fetch_filenames(dn, 1024, wv)
    dates = [dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S") for f in files]
    x = pd.DataFrame(np.array([dates, files]).T, columns=["date", "fname"])
    x["delt"] = np.abs([(u-dn).total_seconds() for u in x.date])
    i = x.delt.idxmin() 
    x = x.iloc[[i]]
    files = [x.fname.tolist()[0]]
    _dict_ = {"date":dn, "resolution": 1024, "wavelength":wv, "loc": "sdo", "verbose": True, "draw": True, "write_id": True,
             "intensity_threshold": th}
    ed = EdgeDetection(files, folder, _dict_).find()
    ed.close()
    if ax == None:
        fig, ax = plt.subplots(dpi=120, figsize=(5, 5))
        set_axes(ax, ed.prob_masked_image, "")
        ax.text(-1024, 1072, "Detected CHB (AIA %d)"%wv, ha="left", va="center", fontdict={"size":10, "color":"r"})
        ax.text(1024, 1072, ed.to_info_str(1), ha="right", va="center", fontdict={"size":10, "color":"b"})
        ax.text(1.03, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":10}, 
                rotation=90, transform=ax.transAxes)
        fig.savefig("data/mva.Figure04.png", bbox_inches="tight")
        os.system("rm -rf data/SDO-Database/*")
    else:
        set_axes(ax, ed.prob_masked_image, "")
        ax.text(-1024, 1072, "Detected CHB (AIA %d)"%wv, ha="left", va="center", fontdict={"size":8, "color":"r"})
        ax.text(1024, 1072, ed.to_info_str(1), ha="right", va="center", fontdict={"size":8, "color":"b"})
        ax.text(1.03, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":10}, 
                rotation=90, transform=ax.transAxes)
    return

def final_image_parameters_masked(dn, ax=None, th=32, wv=193):
    files, folder = fetch_filenames(dn, 1024, wv)
    dates = [dt.datetime.strptime(f.split("_")[0]+f.split("_")[1], "%Y%m%d%H%M%S") for f in files]
    x = pd.DataFrame(np.array([dates, files]).T, columns=["date", "fname"])
    x["delt"] = np.abs([(u-dn).total_seconds() for u in x.date])
    i = x.delt.idxmin() 
    x = x.iloc[[i]]
    files = [x.fname.tolist()[0]]
    _dict_ = {"date":dn, "resolution": 1024, "wavelength":wv, "loc": "sdo", "verbose": True, "draw": True, "write_id": True,
             "intensity_threshold": th}
    ed = EdgeDetection(files, folder, _dict_).find()
    ed.close()
    set_axes(ax, ed.prob_masked_gray_image, "")
    ax.text(-1024, 1096, "AIA %d"%wv, ha="left", va="center", fontdict={"size":8, "color":"r"})
    ax.text(1024, 1096, r"$\mathcal{F}\left(I^C> %d\right)\geq 0.5$"%th, ha="right", va="center", fontdict={"size":8, "color":"k"})
    ax.text(1.05, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":6}, 
                rotation=90, transform=ax.transAxes)
    return

def example_multiple_thresholds(dn, thds):
    fig, axes = plt.subplots(dpi=180, figsize=(3,8), nrows=3, ncols=1)
    for i in range(3):
        final_image_parameters_masked(dn, axes[i], thds[i], 193)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    fig.savefig("data/mva.Figure05.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

def example_multiple_events(dn, thds, wavebands):
    fig, axes = plt.subplots(dpi=180, figsize=(5, 10), nrows=2, ncols=1)
    for i in range(2):
        final_image_parameters(dn, axes[i], thds[i], wavebands[i])        
    fig.subplots_adjust(hspace=0.2, wspace=0.5)
    fig.savefig("data/mva.Figure06.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

if __name__ == "__main__":
    dn = dt.datetime(2018,5,30,12)
    #introduction_image(dn)
    #analysis_image(dn)
    #final_image_parameters(dn)
    #thds = [48, 48]
    #wavebands = [193, 211]
    #example_multiple_events(dt.datetime(2015,9,8,20), thds, wavebands)
    example_multiple_thresholds(dn, [20, 32, 48])
    