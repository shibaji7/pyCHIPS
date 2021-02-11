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

def write(img, txt, blc, fontScale=3, fontColor = (255,255,255), lineType  = 4):
    font = cv2.FONT_HERSHEY_SIMPLEX    
    cv2.putText(img, txt, blc, font, fontScale, fontColor, lineType)
    return

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

def set_plot_axes(ax, image, title, xlabel="X (arcsec)", ylabel="Y (arcsec)", 
             xticks = [-1024,-512,0,512,1024], yticks = [-1024,-512,0,512,1024],
             gray_map=False):
    ax.set_ylabel(ylabel, fontdict={"size":10})
    ax.set_xlabel(xlabel, fontdict={"size":10})
    ax.set_title(title, fontdict={"size":10})
    if gray_map: ax.imshow(image, extent=[-1024,1024,-1024,1024], cmap="gray")
    else: ax.imshow(image, extent=[-1024,1024,-1024,1024])
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if len(xticks)>0: ax.set_xticklabels([r"-$2^{10}$",r"-$2^{9}$","0",r"$2^{9}$",r"$2^{10}$"])
    if len(yticks)>0: ax.set_yticklabels([r"-$2^{10}$",r"-$2^{9}$","0",r"$2^{9}$",r"$2^{10}$"])
    return

def introduction_image(dn):
    conn = get_session()
    wvbins = [193, 211]
    titles = [#"a. AIA 131 (0.4 MK)", "b. AIA 171 (0.7 MK)", 
        "a. AIA 193 (1.2 MK)", "b. AIA 211 (2.0 MK)", 
             #"e. AIA 335 (2.5 MK)", "f. AIA 94 (6.3 MK)"
    ]
    titles = ["(a)", "(b)"]
    fig, axes = plt.subplots(dpi=180, figsize=(6,3), nrows=1, ncols=2)
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
        if wv in [193,211]: 
            image = cv2.rectangle(image, (360,440), (712,712), (255,255,0), 2)
            write(image, titles[j], (100,100))
        rows = image.shape[0]
        image[rows-int(rows/24):rows-1,0:int(rows/2),:] = 0
        ax = axes[j]
        #set_axes(ax, image, titles[j])
        if j==0: set_plot_axes(ax, image, "")
        else: set_plot_axes(ax, image, "", ylabel="", yticks = [], xlabel="", xticks = [])
        os.remove(f)
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
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
    titles = [#"a. Original (AIA 193)", "b. Gray-Scale:HC", 
              #"c. Filtered ($K_{size}=%d$)"%ed.gauss_kernel,
              #"c. Filtered ($K_{size}=%d$):HC"%ed.gauss_kernel, "d. Threshold: Otsu", "f. Conturs", 
              r"e. $\mathcal{F}\left(I^S> %d\right)\geq %.2f$"%(ed.intensity_threshold, ed.intensity_prob_threshold), "f. Detected CHB"]
    titles = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(dpi=180, figsize=(5, 5), nrows=2, ncols=2)
    #set_axes(axes[0,0], cv2.rectangle(ed.org, (360,440), (712,712), (255,255,0), 2), titles[0])
    #set_axes(axes[0,1], ed.gray_hc, titles[1], True)
    #set_axes(axes[1,0], ed.blur, titles[2], True)
    write(ed.blur_mask, titles[0], (100,100))
    print(ed.blur_mask.shape)
    set_plot_axes(axes[0,0], ed.blur_mask, "", gray_map=False, yticks = [], ylabel="", xticks=[], xlabel="")
    write(ed.inv, titles[1], (100,100), fontColor = (0,0,0))
    set_plot_axes(axes[0,1], ed.inv, "", gray_map=True, yticks = [], ylabel="", xticks=[], xlabel="")
    #set_axes(axes[2,1], cv2.cvtColor(ed.src, cv2.COLOR_BGR2RGB), titles[5])
    write(ed.prob_masked_gray_image, titles[2], (100,100))
    set_plot_axes(axes[1,0], ed.prob_masked_gray_image, "", gray_map=True)
    write(ed.prob_masked_image, titles[3], (100,100))
    set_plot_axes(axes[1,1], ed.prob_masked_image, "", yticks = [], ylabel="", xticks=[], xlabel="")
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.savefig("data/mva.Figure03.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

def final_image_parameters(dn, title="", ax=None, th=32, wv=193, wrt=False):
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
        src = cv2.imread(cv2.samples.findFile("data/mva.Figure07.1.png"), cv2.IMREAD_COLOR)
        src = cv2.resize(src, (512, 512))
        src = cv2.pyrUp(src, dstsize=(1024, 1024))
        print(src.shape)
        fig, axes = plt.subplots(dpi=180, figsize=(5, 3), nrows=1, ncols=2)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        write(src, "(a)", (100, 100))
        set_plot_axes(axes[0], src, "")
        
        src = cv2.cvtColor(ed.src, cv2.COLOR_BGR2RGB)
        write(src, "(b)", (100, 100))
        set_plot_axes(axes[1], src, "", yticks = [], ylabel="", xticks=[], xlabel="")
        #set_axes(ax, ed.src, "")
        #ax.text(-1024, 1072, "Detected CHB (AIA %d)"%wv, ha="left", va="center", fontdict={"size":10, "color":"r"})
        #ax.text(1024, 1072, ed.to_info_str(1), ha="right", va="center", fontdict={"size":10, "color":"b"})
        #ax.text(1.03, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":10}, 
        #        rotation=90, transform=ax.transAxes)
        fig.subplots_adjust(hspace=0.01, wspace=0.01)
        fig.savefig("data/mva.Figure04.png", bbox_inches="tight")
        os.system("rm -rf data/SDO-Database/*")
    else:
        write(ed.prob_masked_image, title, (100,100))
        if wrt: set_plot_axes(ax, ed.prob_masked_image, "")
        else: set_plot_axes(ax, ed.prob_masked_image, "", yticks = [], ylabel="", xticks=[], xlabel="")
        #set_axes(ax, ed.prob_masked_image, "")
        #ax.text(-1024, 1072, "Detected CHB (AIA %d)"%wv, ha="left", va="center", fontdict={"size":8, "color":"r"})
        #ax.text(1024, 1072, ed.to_info_str(1), ha="right", va="center", fontdict={"size":8, "color":"b"})
        #ax.text(1.03, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":10}, 
        #        rotation=90, transform=ax.transAxes)
    return

def final_image_parameters_masked(dn, title, ax=None, th=32, wv=193, wrt=False):
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
    #set_axes(ax, ed.prob_masked_gray_image, "")
    write(ed.prob_masked_gray_image, title, (100,100))
    if wrt: set_plot_axes(ax, ed.prob_masked_gray_image, "")
    else: set_plot_axes(ax, ed.prob_masked_gray_image, "", yticks = [], ylabel="", xticks=[], xlabel="")
    #ax.text(-1024, 1096, "AIA %d"%wv, ha="left", va="center", fontdict={"size":8, "color":"r"})
    #ax.text(1024, 1096, r"$\mathcal{F}\left(I^S> %d\right)\geq 0.5$"%th, ha="right", va="center", fontdict={"size":8, "color":"k"})
    #ax.text(1.05, 0.5, dn.strftime("%Y-%m-%d %H:%M:%S UT"), ha="center", va="center", fontdict={"size":6}, 
    #            rotation=90, transform=ax.transAxes)
    return

def example_multiple_thresholds(dn, thds):
    fig, axes = plt.subplots(dpi=180, figsize=(8,3), nrows=1, ncols=3)
    titles = ["(a)", "(b)", "(c)"]
    for i in range(3):
        final_image_parameters_masked(dn, titles[i], axes[i], thds[i], 193, i==0)
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.savefig("data/mva.Figure05.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

def example_multiple_events(dn, thds, wavebands):
    fig, axes = plt.subplots(dpi=180, figsize=(8,3), nrows=1, ncols=3)
    titles = ["(a)", "(b)", "(c)"]
    final_image_parameters(dt.datetime(2018,5,30,12), titles[0], axes[0], wrt=True)
    for i in range(1,3):
        final_image_parameters(dn, titles[i], axes[i], thds[i-1], wavebands[i-1])        
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    fig.savefig("data/mva.Figure06.png", bbox_inches="tight")
    os.system("rm -rf data/SDO-Database/*")
    return

def example_beta():
    from scipy.stats import beta
    x = np.random.beta(7,3,300)
    fig, ax = plt.subplots(dpi=180, figsize=(2.5,2.5), nrows=1, ncols=1)
    u = np.arange(0,1,0.001)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.hist(x, bins=30, color="b", alpha=0.6, histtype="step", density=True)
    ax.plot(u, beta(7,3).pdf(u), "r", ls="-", lw=1.)
    ax.set_xlabel(r"$x_\tau$", fontdict={"size":10})
    ax.set_ylabel("Density", fontdict={"size":10})
    ax.set_xlim(0,1)
    ax.set_ylim(0,5)
    ax.axvline(0.5,color="k", ls="--", lw=0.6)
    u = np.arange(0.5,1,0.001)
    ax.fill_between(u, y1=np.zeros_like(u), y2=beta(7,3).pdf(u), color="r", alpha=0.4)
    ax.text(0.05,0.9,r"$x_\tau\sim$Beta(7,3)",transform=ax.transAxes, fontdict={"size":8, "color":"r"})
    ax.text(0.05,0.8,r"$\mathcal{F}(I^S>I_{th},x_{{\tau}_{th}})=%.1f$"%(np.trapz(beta(7,3).pdf(u), x=u)),
            transform=ax.transAxes, fontdict={"size":8, "color":"r"})
    fig.savefig("data/mva.Figure07.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    dn = dt.datetime(2018,5,30,12)
    #introduction_image(dn)
    #analysis_image(dn)
    #final_image_parameters(dt.datetime(2017,10,26))
    #thds = [48, 48]
    #wavebands = [193, 211]
    #example_multiple_events(dt.datetime(2015,9,8,20), thds, wavebands)
    #example_multiple_thresholds(dn, [20, 32, 48])
    example_beta()
    