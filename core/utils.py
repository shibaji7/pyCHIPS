#!/usr/bin/env python

"""utils.py: Image segnemtation preprocessing """

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import cv2


class BoundaryEstimation(object):
    
    def __init__(self, im, filename, folder, _dict_, cfg_file = "data/config/193.json"):
        self.im = im
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
        if self.conn.chek_remote_file_exists(self.file): 
            self.conn.from_remote_FS(self.file)
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
            self.setup()
        else: raise Exception("File does not exists")
        return

def rect_mask(self, xs, xe, ys, ye, im):
        im[xs:xe,ys:ye,:] = 0
        return im