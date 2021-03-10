"""chips.py: Module is used to implement edge detection tecqniues using CV2"""

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
import json
plt.style.context("seaborn")
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"

import astropy.units as u
from sunpy.net import Fido, attrs
import sunpy.map
from aiapy.calibrate import register, update_pointing, normalize_exposure

import matplotlib.pyplot as plt
import cv2
import os

class RegisterAIA(object):
    
    def __init__(self, date, wavelength=193, resolution=1024, vmin=10):
        self.wavelength = wavelength
        self.date = date
        self.resolution = resolution
        self.vmin = vmin
        self.folder = "data/SDO-Database/{:4d}.{:02d}.{:02d}/{:04d}/{:03d}/".format(self.date.year, self.date.month, self.date.day,
                                                                             self.resolution, self.wavelength)
        self.fname = "{:4d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}.png".format(self.date.year, self.date.month,
                                                                         self.date.day, self.date.hour,
                                                                         self.date.minute, self.date.second)
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        if not os.path.exists(self.folder + self.fname):
            self.normalized()
            self.to_png()
        return
    
    def normalized(self):
        q = Fido.search(
            attrs.Time(self.date.strftime("%Y-%m-%dT%H:%M:%S"), (self.date + dt.timedelta(seconds=11)).strftime("%Y-%m-%dT%H:%M:%S")),
            attrs.Instrument("AIA"),
            attrs.Wavelength(wavemin=self.wavelength*u.angstrom, wavemax=self.wavelength*u.angstrom),
        )
        self.m = sunpy.map.Map(Fido.fetch(q[0,0]))
        m_updated_pointing = update_pointing(self.m)
        m_registered = register(m_updated_pointing)
        self.m_normalized = normalize_exposure(m_registered)
        return
    
    def to_png(self):
        norm, m = self.m_normalized, self.m
        fig, ax = plt.subplots(nrows=1,ncols=1,dpi=100,figsize=(2048/100, 2048/100))
        norm.plot(annotate=False, axes=ax, vmin=self.vmin)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig("tmp.png",bbox_inches="tight")
        im = cv2.imread("tmp.png")
        im = im[10:-10,10:-10]
        im = cv2.resize(im, (self.resolution, self.resolution))
        os.remove("tmp.png")
        cv2.imwrite(self.folder + self.fname, im)
        return

class Chips(object):
    """ Edge detection by Open-CV """
    
    def __init__(self, filename, folder, _dict_, cfg_file = "data/config/{:3d}.json"):
        cfg_file = cfg_file.format(_dict_["wavelength"])
        with open(cfg_file, "r") as fp:
            dic = json.load(fp)
            for p in dic.keys():
                if not hasattr(self, p): setattr(self, p, dic[p])
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self.filename = filename
        self.folder = folder
        self.file = folder + filename
        self.write_id_index = 0
        self.contour_infos = {}
        print(" Proc file - ", self.file)
        if os.path.exists(self.file): 
            self.extn = "." + filename.split(".")[-1]
            self.images = {}
            self.images["src"] = cv2.imread(cv2.samples.findFile(self.file), cv2.IMREAD_COLOR)
            self.images["src"] = self.rect_mask(self.resolution-int(self.resolution/24), self.resolution-1, 
                                                0, int(self.resolution/2), self.images["src"])
            self.img_params = {"resolution": self.images["src"].shape[0]}
            self.images["src"] = self.rescale(self.images["src"], self.resolution)
            self.images["org"] = np.copy(cv2.cvtColor(self.images["src"], cv2.COLOR_BGR2RGB))            
            self.images["prob_masked_gray_image"], self.images["prob_masked_image"] =\
                        np.zeros_like(self.images["src"]), np.copy(self.images["src"])
            self.images["gray"] = cv2.cvtColor(self.images["src"], cv2.COLOR_BGR2GRAY)
            self.images["contours"] = np.zeros_like(self.images["gray"])
            self.images["contour_maps"] = np.zeros_like(self.images["gray"])
            self.images["bin_map"] = np.zeros_like(self.images["gray"])
            self.setup()
        else: raise Exception("File does not exists")
        return
    
    def rescale(self, img, to):
        """ Resacle the images """
        dsize = (to, to)
        img = cv2.resize(img, dsize)
        return img
    
    def setup(self):
        """ Load parameters """
        mult = int(self.resolution/512) + 0.5
        self.prims = np.array(self.prims) * mult
        self.delta = int(self.resolution/128)
        return
    
    def run(self):
        self.detect_hough_circles()
        self.detect_edges()
        return
    
    def get_center(self, contour):
        M = cv2.moments(contour)
        center = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        return center
    
    def distance(self, contour):
        d = self.radius
        try:
            center = self.get_center(contour)
            d = np.sqrt((self.center[0]-center[0])**2+(self.center[1]-center[1])**2)
        except: print(" distance: Div by (0)")
        return d
    
    def rect_mask(self, xs, xe, ys, ye, im):
        im[xs:xe,ys:ye,:] = 0
        return im
    
    def detect_hough_circles(self):
        """ Detecting the Hough Circles """
        gray = np.copy(self.images["gray"])
        rows = self.resolution
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=self.hc_param1, param2=self.hc_param2,
                                   minRadius=int(rows/3), maxRadius=int(rows/2))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            radi = int(rows/2)
            for i in circles[0, :]:
                if radi - self.alpha <= i[0] <= radi + self.alpha and radi - self.alpha <= i[1] <= radi + self.alpha:
                    self.center = (i[0], i[1])
                    self.radius = i[2] + self.delta
        return
    
    def detect_edges(self):
        """ Detect edges in the images """
        rows = self.resolution
        src = np.copy(self.images["src"])        
        gray = np.copy(self.images["gray"])
        gray_hc = np.copy(gray)
        cv2.circle(gray_hc, self.center, 1, (255, 255, 255), 3)
        cv2.circle(gray_hc, self.center, self.radius, (255, 255, 255), 3)
        cv2.circle(self.images["prob_masked_gray_image"], self.center, self.radius, (255, 255, 255), 3)
        blur = cv2.GaussianBlur(gray, (self.gauss_kernel, self.gauss_kernel), 0)
        mask = np.zeros_like(gray)
        cv2.circle(mask, self.center, self.radius, tuple(self.draw_param["hc_circle"]["color"]), -1)
        mask[mask > 0] = 1
        blur_mask = mask*blur
        if not self.advt: _, thrs = cv2.threshold(blur_mask, self.ed_th_low, self.ed_th_high, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else: thrs = cv2.adaptiveThreshold(blur_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
        _, inv = cv2.threshold(blur_mask, self.ed_th_low, self.ed_th_high, cv2.THRESH_BINARY_INV)
        inv = 255 - thrs
        cv2.circle(inv, self.center, self.radius-int(self.delta/2), (0,0,0), 20)
        self.contours, self.hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        perimeters = [cv2.arcLength(self.contours[i],True) for i in range(len(self.contours))]
        x = pd.DataFrame(np.array([perimeters, range(len(perimeters))]).T, columns=["perims", "ids"])
        x = x.sort_values("perims",ascending=False)
        ix = 0
        for idx, perimeter in zip(x.ids, x.perims):
            contour = self.contours[int(idx)]
            if perimeter > self.prims[0] and perimeter < self.prims[1] and self.distance(contour) < self.radius: 
                print(" Arc Len:", perimeter)
                self.image_intensity(gray, idx)
                self.draw_contour(contour, src)
                self.rec_depth, self.parent = 0, int(ix)
                self.draw_nested_contour(src, self.hierarchy[0, int(idx), 2])
                ix += 1
        self.images["src_olc"] = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.images["gray_hc"] = gray_hc
        self.images["inv"] = inv
        self.images["blur_mask"] = blur_mask
        self.images["blur"] = blur
        self.images["prob_masked_image"] = cv2.cvtColor(self.images["prob_masked_image"], cv2.COLOR_BGR2RGB)
        self.analysis_plot()
        self.final_image()
        self.masked_image()
        return
    
    def draw_contour(self, contour, img, gray=False, col=None):
        if col is None: col=tuple(self.draw_param["contur"]["color"])
        for _x in range(contour.shape[0]-1):
            if gray: cv2.line(img, (contour[_x,0,0], contour[_x,0,1]), (contour[_x+1,0,0], contour[_x+1,0,1]), 
                              (255,255,255), self.draw_param["contur"]["thick"])                              
            else: cv2.line(img, (contour[_x,0,0], contour[_x,0,1]), (contour[_x+1,0,0], contour[_x+1,0,1]), 
                     col, self.draw_param["contur"]["thick"])
        if gray: cv2.line(img, (contour[0,0,0], contour[0,0,1]), (contour[-1,0,0], contour[-1,0,1]), 
                 (255,255,255), self.draw_param["contur"]["thick"])
        else: cv2.line(img, (contour[0,0,0], contour[0,0,1]), (contour[-1,0,0], contour[-1,0,1]), 
                 col, self.draw_param["contur"]["thick"])
        return
    
    def draw_nested_contour(self, img, fc_np, gray=False):
        # [Next, Previous, First_Child, Parent]
        for i, pnt in enumerate(self.hierarchy[0,:,3]):
            if pnt==self.parent: self.draw_contour(self.contours[i], img, gray, tuple(self.draw_param["contur"]["nest_color"]))
        return
    
    def draw_bin_map(self, img, ix, col=(255, 255, 255)):
        cv2.drawContours(img, self.contours, ix, col, thickness=-1)
        return
    
    def draw_nested_bin_map(self, img):
        for i, pnt in enumerate(self.hierarchy[0,:,3]):
            if pnt==self.parent: self.draw_bin_map(img, i, col=(0,0,0))
        return
    
    def image_intensity(self, gray, ix):
        cimg = np.zeros_like(gray)
        cv2.drawContours(cimg, self.contours, int(ix), color=255, thickness=-1)
        pts = np.where(cimg == 255)
        intensity = self.intensity_threshold-gray[pts[0], pts[1]].ravel().astype(int)
        prob = 1-np.median(1./(1.+np.exp(intensity)))
        col = np.array([255,255,255])*prob
        cv2.drawContours(self.images["prob_masked_gray_image"], self.contours, int(ix), tuple(col), thickness=-1)
        if prob >= self.intensity_prob_threshold:
            self.write_id_index += 1
            self.rec_depth, self.parent = 0, int(ix)
            self.draw_contour(self.contours[int(ix)], self.images["prob_masked_image"])
            self.draw_nested_contour(self.images["prob_masked_image"], self.hierarchy[0, int(ix), 2])
            self.rec_depth, self.parent = 0, int(ix)
            self.draw_contour(self.contours[int(ix)], self.images["contours"], True)
            self.draw_nested_contour(self.images["contours"], self.hierarchy[0, int(ix), 2], True)
            self.rec_depth, self.parent = 0, int(ix)
            self.draw_bin_map(self.images["bin_map"], int(ix))
            self.draw_nested_bin_map(self.images["bin_map"])
            if hasattr(self, "write_id") and self.write_id: 
                self.write(self.images["prob_masked_image"], "CH:%d"%self.write_id_index, self.get_center(self.contours[int(ix)]))
                self.extract_informations(self.contours[int(ix)])
        return prob
    
    def extract_informations(self, contour):
        self.contour_infos[self.write_id_index] = {}
        circle = np.zeros_like(self.images["gray"])
        cv2.circle(circle, self.center, self.radius-self.delta, (255,255,255), -1)
        _, thrs = cv2.threshold(circle, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cntrs, _ = cv2.findContours(thrs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c_area = cv2.contourArea(cntrs[0])
        c_perim = cv2.arcLength(cntrs[0],True)
        self.contour_infos[self.write_id_index]["area"] = np.round(cv2.contourArea(contour)/(c_area),2)
        self.contour_infos[self.write_id_index]["d"] = np.round(self.distance(contour)/self.resolution,2)
        self.contour_infos[self.write_id_index]["perim"] = np.round(cv2.arcLength(contour,True)/c_perim,2)
        return
    
    def to_info_str(self, ix):
        st = ""
        for i in range(1,ix+1):
            st = st + f"$\Delta A_{i}$={self.contour_infos[i]['area']}$A_c$, $d^c_{i}$={self.contour_infos[i]['d']}R" + "\n"
        st = st[:-1]
        return st
    
    def write(self, img, txt, blc):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.
        fontColor = (255,127,0)
        lineType  = 2
        cv2.putText(img, txt, blc, font, fontScale, fontColor, lineType)
        return
    
    def analysis_plot(self):
        """ Steps of the analysis plots """
        titles = ["a. Original (AIA 193)", "b. Gray-Scale:HC", "c. Filtered ($K_{size}=%d$)"%self.gauss_kernel,
                  "d. Masked:HC", "e. Threshold: OTSU", "f. Conturs", 
                  r"g. $\mathcal{F}\left(I^C> %d\right)\geq %.2f$"%(self.intensity_threshold, self.intensity_prob_threshold), 
                  "h. Detected CHB"]
        fig, axes = plt.subplots(dpi=180, figsize=(5, 10), nrows=4, ncols=2)
        self.set_axes(axes[0,0], self.images["org"], titles[0])
        self.set_axes(axes[0,1], self.images["gray_hc"], titles[1], True)
        self.set_axes(axes[1,0], self.images["blur"], titles[2], True)
        self.set_axes(axes[1,1], self.images["blur_mask"], titles[3], True)
        self.set_axes(axes[2,0], self.images["inv"], titles[4], True)
        self.set_axes(axes[2,1], self.images["src_olc"], titles[5])
        self.set_axes(axes[3,0], self.images["prob_masked_gray_image"], titles[6], True)
        self.set_axes(axes[3,1], self.images["prob_masked_image"], titles[7])
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.savefig(self.folder + self.filename.replace(self.extn, "_analysis" + self.extn), bbox_inches="tight")
        return
    
    def set_axes(self, ax, image, title, gray_map=False):
        ax.set_ylabel("Y (arcsec)", fontdict={"size":10})
        ax.set_xlabel("X (arcsec)", fontdict={"size":10})
        ax.set_title(title, fontdict={"size":10})
        if gray_map: ax.imshow(image, extent=[-1024,1024,-1024,1024], cmap="gray")
        else: ax.imshow(image, extent=[-1024,1024,-1024,1024])
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xticks([-1024,-512,0,512,1024])
        ax.set_yticks([-1024,-512,0,512,1024])
        return
    
    def masked_image(self):
        fig, ax = plt.subplots(dpi=120, figsize=(5, 5))
        self.set_axes(ax, self.rescale(self.images["contours"], 4096), "", True)
        ax.text(-1024, 1072, "Detected CHB (AIA %d)"%self.wavelength, ha="left", va="center", fontdict={"size":10, "color":"r"})
        if self.write_id: ax.text(1024, 1072, self.to_info_str(1), ha="right", va="center", fontdict={"size":10, "color":"b"})
        ax.text(1.03, 0.5, self.date.strftime("%Y-%m-%d %H:%M:%S UT"), 
                ha="center", va="center", fontdict={"size":10}, rotation=90, transform=ax.transAxes)
        fig.savefig(self.folder + self.filename.replace(self.extn, "_contours" + self.extn), bbox_inches="tight")
        binmaps = np.copy(self.images["bin_map"])
        cv2.circle(binmaps, self.center, self.radius, (0,0,0), 3)
        binmaps = self.rescale(binmaps, 4096)
        self.fcontours = (binmaps > 0).astype(int)
        np.savetxt(self.folder + self.filename.replace(self.extn, "_binmaps.txt"), self.fcontours, fmt="%i")
        cv2.imwrite(self.folder + self.filename.replace(self.extn, "_binmaps" + self.extn), self.fcontours*255)
        return
    
    def final_image(self):
        fig, ax = plt.subplots(dpi=120, figsize=(5, 5))
        self.set_axes(ax, self.rescale(self.images["prob_masked_image"], 4096), "")
        ax.text(-1024, 1072, "Detected CHB (AIA %d)"%self.wavelength, ha="left", va="center", fontdict={"size":10, "color":"r"})
        if self.write_id: ax.text(1024, 1072, self.to_info_str(1), ha="right", va="center", fontdict={"size":10, "color":"b"})
        ax.text(1.03, 0.5, self.date.strftime("%Y-%m-%d %H:%M:%S UT"), 
                ha="center", va="center", fontdict={"size":10}, rotation=90, transform=ax.transAxes)
        fig.savefig(self.folder + self.filename.replace(self.extn, "_fout" + self.extn), bbox_inches="tight")
        return