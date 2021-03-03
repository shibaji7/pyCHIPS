"""cv2_edge_detectors.py: Module is used to implement different edge detection tecqniues using CV2"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import os
import numpy as np
import pandas as pd
import cv2
import json
from to_remote import get_session

from data_pipeline import fetch_filenames

class EdgeDetection(object):
    """ Edge detection by Open-CV """
    
    def __init__(self, _files_, _dir_, _dict_):
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self._files_ = _files_
        self.folder = _dir_
        self.conn = get_session()
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.image = np.zeros((self.resolution, self.resolution, 3), np.uint8)
        self.setup()
        self.write_id_index = 0
        self.contour_infos = {}
        return
    
    def setup(self):
        """ Load parameters """
        mult = int(self.resolution/512) + 0.5
        N = 4
        for i in range(1,N):
            cv2.line(self.image, (int(self.resolution/N)*i, 0), (int(self.resolution/N)*i, self.resolution), (255, 255, 255), thickness=1)
            cv2.line(self.image, (0, int(self.resolution/N)*i), (self.resolution, int(self.resolution/N)*i), (255, 255, 255), thickness=1)
        with open("data/config/config_cv2.json", "r") as fp:
            dic = json.load(fp)
            for p in dic[str(self.wavelength)].keys():
                if not hasattr(self, p): setattr(self, p, dic[str(self.wavelength)][p])
        self.prims = np.array(self.prims) * mult
        self.delta = int(self.resolution/128)
        return
    
    def find(self):
        """ Find contours by CV2 """
        self._files_.sort()
        for fname in self._files_:
            f = self.folder+fname
            print(" Proc File:",f)
            #if self.conn.chek_remote_file_exists(f): 
            #    self.conn.from_remote_FS(f)
            self.detect_hough_circles(f)
            self.detect_edges(f)
            #break
        return self
    
    def close(self):
        self.conn.close()
        #if os.path.exists(self.folder): 
        #    for fname in self._files_:
        #        if os.path.exists(self.folder+fname): os.remove(self.folder+fname)
        return
    
    def detect_hough_circles(self, f):
        """ Detecting the Hough Circles """
        src = cv2.imread(cv2.samples.findFile(f), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=self.hc_param1, param2=self.hc_param2,
                                   minRadius=int(rows/3), maxRadius=int(rows/2))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            radi = int(rows/2)
            for i in circles[0, :]:
                if radi - self.alpha <= i[0] <= radi + self.alpha and radi - self.alpha <= i[1] <= radi + self.alpha:
                    self.center = (i[0], i[1])
                    cv2.circle(src, self.center, 1, tuple(self.draw_param["hc_center"]["color"]), 
                               self.draw_param["hc_center"]["thick"])# circle center
                    cv2.circle(self.image, self.center, 1, tuple(self.draw_param["hc_center"]["color"]), 
                               self.draw_param["hc_center"]["thick"])
                    self.radius = i[2] + self.delta
                    cv2.circle(src, self.center, self.radius, tuple(self.draw_param["hc_circle"]["color"]), 
                               self.draw_param["hc_circle"]["thick"])# circle outline
                    cv2.circle(self.image, self.center, self.radius, tuple(self.draw_param["hc_circle"]["color"]), 
                               self.draw_param["hc_circle"]["thick"])
        #if self.draw: cv2.imwrite(f.replace(".jpg", "_hc.jpg"), src)
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
    
    def detect_edges(self, f):
        """ Detect edges in the images """
        self.src = cv2.imread(cv2.samples.findFile(f), cv2.IMREAD_COLOR)
        rows = len(self.src)
        self.src = self.rect_mask(rows-int(rows/24), rows-1, 0, int(rows/2), self.src)
        self.org = self.rect_mask(rows-int(rows/24), rows-1, 0, int(rows/2),
                                  cv2.cvtColor(cv2.imread(cv2.samples.findFile(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        self.prob_masked_gray_image, self.prob_masked_image = np.zeros_like(self.src), np.copy(self.org)
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        self.gray_hc = np.copy(self.gray)
        cv2.circle(self.gray_hc, self.center, 1, (255, 255, 255), self.draw_param["hc_circle"]["thick"])
        cv2.circle(self.gray_hc, self.center, self.radius, (255, 255, 255), self.draw_param["hc_circle"]["thick"])
        cv2.circle(self.prob_masked_gray_image, self.center, self.radius, (255, 255, 255), self.draw_param["hc_circle"]["thick"])
        self.blur = cv2.GaussianBlur(self.gray, (self.gauss_kernel, self.gauss_kernel), 0)
        self.mask = np.zeros_like(self.gray)
        cv2.circle(self.mask, self.center, self.radius, tuple(self.draw_param["hc_circle"]["color"]), -1)
        self.mask[self.mask > 0] = 1
        self.blur_mask = self.mask*self.blur
        #cv2.circle(self.blur_mask, self.center, self.radius, (255, 255, 255), self.draw_param["hc_circle"]["thick"])
        if not self.advt: _, thrs = cv2.threshold(self.blur_mask, self.ed_th_low, self.ed_th_high, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else: thrs = cv2.adaptiveThreshold(self.blur_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.inv = 255 - thrs
        cv2.circle(self.inv, self.center, self.radius-int(self.delta/2), (0,0,0), self.draw_param["hc_circle"]["thick"])
        self.contours, self.hierarchy = cv2.findContours(self.inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        perimetes = [cv2.arcLength(self.contours[i],True) for i in range(len(self.contours))]
        x = pd.DataFrame(np.array([perimetes, range(len(perimetes))]).T, columns=["perimetes", "ids"])
        x = x.sort_values("perimetes",ascending=False)
        ix = 0
        for idx, perimeter in zip(x.ids, x.perimetes):
            contour = self.contours[int(idx)]
            if perimeter > (2*np.pi*self.radius)+100 or perimeter < (2*np.pi*self.radius)-100:
                print(" Arc perim:", perimeter)
                self.image_intensity(self.gray, idx)
                self.draw_contour(contour, self.src)
                self.rec_depth = 0
                self.draw_nested_contour(self.src, self.hierarchy[0, int(idx), 2])
                self.draw_contour(contour, self.image)
                ix += 1
        #cv2.imwrite(f.replace(".jpg", "_overlaied.jpg"), self.src)
        #cv2.imwrite(f.replace(".jpg", "_contours.jpg"), self.image)
        self.blur_mask = cv2.cvtColor(self.blur_mask,cv2.COLOR_GRAY2RGB)
        cv2.circle(self.blur_mask, self.center, self.radius, (255,0,0),5)
        return
    
    def draw_contour(self, contour, img, col=None):
        perim = cv2.arcLength(contour, True)
        if perim > (2*np.pi*self.radius)+500 or perim < (2*np.pi*self.radius)-100:
            if col is None: col=tuple(self.draw_param["contur"]["color"])
            for _x in range(contour.shape[0]-1):
                cv2.line(img, (contour[_x,0,0], contour[_x,0,1]), (contour[_x+1,0,0], contour[_x+1,0,1]), 
                         col, self.draw_param["contur"]["thick"])
            cv2.line(img, (contour[0,0,0], contour[0,0,1]), (contour[-1,0,0], contour[-1,0,1]), 
                     col, self.draw_param["contur"]["thick"])
        return
    
    def draw_nested_contour(self, img, fc_np):
        # [Next, Previous, First_Child, Parent]
        rd = 50
        self.rec_depth += 1
        self.draw_contour(self.contours[fc_np], img, tuple(self.draw_param["contur"]["nest_color"]))
        if self.hierarchy[0, fc_np, 0] > 0 and self.rec_depth < rd: self.draw_nested_contour(img, self.hierarchy[0, fc_np, 0])
        if self.hierarchy[0, fc_np, 1] > 0 and self.rec_depth < rd: self.draw_nested_contour(img, self.hierarchy[0, fc_np, 1])
        return
    
    def image_intensity(self, gray, ix):
        cimg = np.zeros_like(gray)
        cv2.drawContours(cimg, self.contours, int(ix), color=255, thickness=-1)
        pts = np.where(cimg == 255)
        intensity = self.intensity_threshold-gray[pts[0], pts[1]].ravel().astype(int)
        prob = 1-np.median(1./(1.+np.exp(intensity)))
        col = np.array([255,255,255])*prob
        #cv2.drawContours(self.prob_masked_gray_image, self.contours, int(ix), tuple(col), thickness=-1)
        if prob >= self.intensity_prob_threshold:
            cv2.drawContours(self.prob_masked_gray_image, self.contours, int(ix), tuple(col), thickness=-1)
            self.write_id_index += 1
            self.rec_depth = 0
            self.draw_contour(self.contours[int(ix)], self.prob_masked_image)
            self.draw_nested_contour(self.prob_masked_image, self.hierarchy[0, int(ix), 2])
            if hasattr(self, "write_id") and self.write_id: 
                if self.write_id_index==1:
                    self.write(self.prob_masked_image, "CH:%d"%self.write_id_index, self.get_center(self.contours[int(ix)]))
                self.extract_informations(self.contours[int(ix)])
        return prob
    
    def write(self, img, txt, blc):
        blc = (blc[0]+64, blc[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2.
        fontColor = (255,255,32)
        lineType  = 3
        cv2.putText(img, txt, blc, font, fontScale, fontColor, lineType)
        return
    
    def extract_informations(self, contour):
        self.contour_infos[self.write_id_index] = {}
        circle = np.zeros_like(self.gray)
        cv2.circle(circle, self.center, self.radius-self.delta, (255,255,255), -1)
        _, thrs = cv2.threshold(circle, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cntrs, _ = cv2.findContours(thrs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(cntrs)
        c_area = 2*np.pi*self.radius**2#cv2.contourArea(cntrs[0],False)
        c_perim = cv2.arcLength(cntrs[0],True)
        self.contour_infos[self.write_id_index]["area"] = np.round(cv2.contourArea(contour)/(c_area),2)
        self.contour_infos[self.write_id_index]["d"] = np.round(self.distance(contour)/self.radius,2)
        self.contour_infos[self.write_id_index]["perim"] = np.round(cv2.arcLength(contour,True)/c_perim,2)
        
        if self.contour_infos[self.write_id_index]["area"] > 1: 
            print(self.contour_infos[self.write_id_index]["area"], np.round(self.resolution**2/(c_area),2))
            self.contour_infos[self.write_id_index]["area"] = np.round(cv2.contourArea(contour)/(c_area),2) -\
            np.round(self.resolution**2/(c_area),2)
        return
    
    def to_info_str(self, ix):
        st = ""
        for i in range(1,ix+1):
            st = st + f"$\Delta A_{i}$={self.contour_infos[i]['area']}$A_c$, $d^c_{i}$={self.contour_infos[i]['d']}R" + "\n"
        st = st[:-1]
        return st