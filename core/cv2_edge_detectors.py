"""cv2_edge_detectors.py: Module is used to implement different edge detection tecqniues using CV2"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import numpy as np
import pandas as pd
import cv2
from to_remote import get_session

class EdgeDetection(object):
    """ Edge detection by Open-CV """
    
    def __init__(self, _files_, _dir_, _dict_):
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self._files_ = _files_
        self.folder = _dir_
        self.conn = get_session()
        if not os.path.exists(self.folder): os.system("mkdir -p " + self.folder)
        self.setup()
        return
    
    def setup(self):
        """ Load parameters """
        self.alpha = 10
        self.gauss_kernel = 13
        self.prims = [200., 1000.]
        self.iter = 2
        return
    
    def find(self):
        """ Find contours by CV2 """
        self._files_.sort()
        for fname in self._files_:
            f = self.folder+fname
            print(" Proc File:",f)
            if self.conn.chek_remote_file_exists(f): 
                self.conn.from_remote_FS(f)
                self.detect_hough_circles(f)
                self.detect_edges(f)
            break
        return self
    
    def close(self):
        self.conn.close()
        if os.path.exists(self.folder): 
            for fname in self._files_:
                if os.path.exists(self.folder+fname): os.remove(self.folder+fname)
        return
    
    def detect_hough_circles(self, f):
        """ Detecting the Hough Circles """
        src = cv2.imread(cv2.samples.findFile(f), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=int(rows/3), maxRadius=int(rows/2))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            radi = int(rows/2)
            for i in circles[0, :]:
                if radi - self.alpha <= i[0] <= radi + self.alpha and radi - self.alpha <= i[1] <= radi + self.alpha:
                    self.center = (i[0], i[1])
                    cv2.circle(src, self.center, 1, (0, 255, 0), 2)# circle center
                    self.radius = i[2]
                    cv2.circle(src, self.center, self.radius, (255, 0, 255), 1)# circle outline
        if self.draw: cv2.imwrite(f.replace(".jpg", "_hc.jpg"), src)
        return
    
    def distance(self, contour):
        M = cv2.moments(contour)
        center = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        d = np.sqrt((self.center[0]-center[0])**2+(self.center[1]-center[1])**2)
        return d
    
    def detect_edges(self, f):
        """ Detect edges in the images """
        src = cv2.imread(cv2.samples.findFile(f), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.gauss_kernel, self.gauss_kernel), 0)
        _, thrs = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        inv = 255 - thrs
        contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
        x = pd.DataFrame(np.array([perimeters, range(len(perimeters))]).T, columns=["perims", "ids"])
        x = x.sort_values("perims",ascending=False)
        ix = 0
        for idx, perimeter in zip(x.ids, x.perims):
            contour = contours[int(idx)]            
            if perimeter > self.prims[0] and perimeter < self.prims[1] and self.distance(contour) < self.radius: 
                print(" Arc Len:", perimeter)
                for _x in range(contour.shape[0]-1):
                    cv2.line(src, (contour[_x,0,0], contour[_x,0,1]),
                             (contour[_x+1,0,0], contour[_x+1,0,1]), (0,255,0), 1)
                cv2.line(src, (contour[0,0,0], contour[0,0,1]),
                         (contour[-1,0,0], contour[-1,0,1]), (0,255,0), 1)
                ix += 1
                if ix == self.iter: break
        cv2.imwrite(f.replace(".jpg", "_contours.jpg"), src)
        return