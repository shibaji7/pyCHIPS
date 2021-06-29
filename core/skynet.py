#!/usr/bin/env python

"""skynet.py: Image segnemtation algoritm for Solar Disk: SkyNet"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib.pyplot as plt
import matplotlib as mpl    
from dateutil import parser as prs
import pandas as pd
import datetime as dt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
import os
import json

_params_ = {
    "_desc": "PyTorch Unsupervised Segmentation",
    "nChannel": "Number of channels",
    "maxIter": "Number of maximum iterations",
    "minLabels": "Minimum number of labels",
    "lr": "Learning rate",
    "nConv": "Number of convolutional layers",
    "nConv": "Number of convolutional layers",
    "stepsize_sim": "Step size for similarity loss",
    "stepsize_con": "Step size for continuity loss",
}

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

# CNN model
class SkyNet(nn.Module):
    
    def __init__(self, input_dim, params):
        super(SkyNet, self).__init__()
        self.params = params
        self.conv1 = nn.Conv2d(input_dim, self.params.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(self.params.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(self.params.nChannel, self.params.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(self.params.nChannel) )
        self.conv3 = nn.Conv2d(self.params.nChannel, self.params.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(self.params.nChannel)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.params.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class Loader(object):
    
    def __init__(self, fname, folder, date, params, save=True, cfg_file = "data/config/{:3d}.json"):
        self.fname = fname
        self.folder = folder
        self.setup(cfg_file, params)
        self.date = date        
        self.params = params
        self.resolution = params.resolution
        self.wavelength = params.wavelength
        self.save = save
        self.extn = "." + fname.split(".")[-1]
        self.use_cuda = torch.cuda.is_available()
        self.detect_hough_circles()
        self.im = cv2.imread(self.folder + self.fname)
        self.data = torch.from_numpy( np.array([self.im.transpose( (2, 0, 1) ).astype("float32")/255.]) )
        if self.use_cuda: self.data = self.data.cuda()
        self.data = Variable(self.data)
        return
    
    def rescale(self, img, to):
        """ Resacle the images """
        dsize = (to, to)
        img = cv2.resize(img, dsize)
        return img
    
    def setup(self, cfg_file, params):
        """ Load parameters """
        _dict_ = {}
        for k in vars(params).keys():
            _dict_[k] = vars(args)[k]
        cfg_file = cfg_file.format(_dict_["wavelength"])
        with open(cfg_file, "r") as fp:
            dic = json.load(fp)
            for p in dic.keys():
                if not hasattr(self, p): setattr(self, p, dic[p])
        for p in _dict_.keys():
            setattr(self, p, _dict_[p])
        self._dict_ = _dict_
        mult = int(self.resolution/512) + 0.5
        self.prims = np.array(self.prims) * mult
        self.delta = int(self.resolution/128)
        
        self.images = {}
        self.images["src"] = cv2.imread(cv2.samples.findFile(self.folder+self.fname), cv2.IMREAD_COLOR)
        self.img_params = {"resolution": self.images["src"].shape[0]}
        self.images["src"] = self.rescale(self.images["src"], self.resolution)
        self.images["org"] = np.copy(cv2.cvtColor(self.images["src"], cv2.COLOR_BGR2RGB))            
        self.images["prob_masked_gray_image"], self.images["prob_masked_image"] =\
        np.zeros_like(self.images["src"]), np.copy(self.images["src"])
        self.images["gray"] = cv2.cvtColor(self.images["src"], cv2.COLOR_BGR2GRAY)
        self.images["contours"] = np.zeros_like(self.images["gray"])
        self.images["contour_maps"] = np.zeros_like(self.images["gray"])
        self.images["bin_map"] = np.zeros_like(self.images["gray"])
        self.images["hc_mask"] = np.zeros_like(self.images["gray"])
        self.images["prob_masked_image_ol"] = np.copy(self.images["src"])
        return
    
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
                    self.radius = i[2]
                    self.create_hc_mask()
        return
    
    def create_hc_mask(self):
        # HC mask create
        mask = cv2.circle(self.images["hc_mask"], self.center, self.radius, 255, -1)
        self.images["hc_mask"] = mask
        self.NzC = np.count_nonzero(mask)
        return
    
    def load_model(self):
        # Load & train model
        self.model = SkyNet( self.data.size(1), self.params)
        if self.use_cuda: self.model.cuda()
        self.model.train()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()
        # continuity loss definition
        self.loss_hpy = torch.nn.L1Loss(size_average = True)
        self.loss_hpz = torch.nn.L1Loss(size_average = True)

        self.HPy_target = torch.zeros(self.im.shape[0]-1, self.im.shape[1], self.params.nChannel)
        self.HPz_target = torch.zeros(self.im.shape[0], self.im.shape[1]-1, self.params.nChannel)
        if self.use_cuda: self.HPy_target, self.HPz_target = self.HPy_target.cuda(), self.HPz_target.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        self.label_colours = np.random.randint(255, size=(100,3))
        return self
    
    def run_forwarding(self):
        if not os.path.exists(self.folder + self.fname.replace(self.extn, "_seg" + self.extn)):
            for batch_idx in range(self.params.maxIter):
                self.optimizer.zero_grad()
                self.output = self.model( self.data )[ 0 ]
                self.output = self.output.permute( 1, 2, 0 ).contiguous().view( -1, self.params.nChannel )

                self.outputHP = self.output.reshape( (self.im.shape[0], self.im.shape[1], self.params.nChannel) )
                self.HPy = self.outputHP[1:, :, :] - self.outputHP[0:-1, :, :]
                self.HPz = self.outputHP[:, 1:, :] - self.outputHP[:, 0:-1, :]
                self.lhpy = self.loss_hpy(self.HPy, self.HPy_target)
                self.lhpz = self.loss_hpz(self.HPz, self.HPz_target)

                ignore, target = torch.max( self.output, 1 )
                im_target = target.data.cpu().numpy()
                nLabels = len(np.unique(im_target))
                self.loss = self.params.stepsize_sim * self.loss_fn(self.output, target) + self.params.stepsize_con * (self.lhpy + self.lhpz)
                self.loss.backward()
                self.optimizer.step()
                print (batch_idx, "/", self.params.maxIter, "|", " label num :", nLabels, " | loss : %.2f"%self.loss.item())
                if nLabels <= self.params.minLabels or self.params.los > self.loss: 
                    print (" >> nLabels:", nLabels, " reached minLabels:", self.params.minLabels, " with loss: %.2f."%self.loss.item())
                    break
        return self
    
    def save_outputs(self):
        if not os.path.exists(self.folder + self.fname.replace(self.extn, "_seg" + self.extn)):
            output = self.model(self.data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, self.params.nChannel)
            ignore, target = torch.max(output, 1)
            self.im_target = target.data.cpu().numpy()
            self.unique_targets = np.array([self.label_colours[c % 100] for c in np.unique(self.im_target)])
            self.im_target_rgb = np.array([self.label_colours[c % 100] for c in self.im_target])
            self.im_target_rgb = self.im_target_rgb.reshape(self.im.shape).astype(np.uint8)
            cv2.imwrite(self.folder + self.fname.replace(self.extn, "_seg" + self.extn), self.im_target_rgb)
        return self
    
    def check_intensity(self, mask):
        # Check the intensity bound on the output
        ints = False
        gray = np.copy(self.images["gray"])
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        if np.quantile(gray.ravel(), 0.05) >= 0 and np.quantile(gray.ravel(), 0.95) <= 100: ints = True
        return ints
    
    def estimate_CHB(self):
        if not os.path.exists(self.folder + self.fname.replace(self.extn, "_msk" + self.extn)):
            maskList = []
            for targ in self.unique_targets:
                mask = cv2.inRange(self.im_target_rgb, targ-1, targ+1)
                mask = cv2.bitwise_and(mask, mask, mask=self.images["hc_mask"])
                if np.count_nonzero(mask) < 0.1*self.NzC and self.check_intensity(mask):
                    maskList.append(mask)
            self.totalmask = np.array(maskList).sum(axis=0)
            cv2.imwrite(self.folder + self.fname.replace(self.extn, "_msk" + self.extn), self.totalmask)
        else:
            fname = self.folder + self.fname.replace(self.extn, "_msk" + self.extn)
            self.totalmask = cv2.cvtColor(cv2.imread(fname, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        return self
    
    def rescale_prob(self, prob, l=0., u=0.5):
        prob = prob*(u-l) + l
        return prob
    
    def calculate_probabilistic_boundaries(self):
        self.contours, self.hierarchy = cv2.findContours((255-self.totalmask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for ix in range(len(self.contours)):
            if ix > 0: self.draw_contour(self.contours[int(ix)], self.images["prob_masked_image"])
        cv2.imwrite(self.folder + self.fname.replace(self.extn, "_stg0.0_out" + self.extn), self.images["prob_masked_image"])
        gray, mask = np.copy(self.images["gray"]), np.zeros_like(self.images["gray"])
        contours = self.contours[1:]
        cv2.drawContours(mask, contours, -1, 255, -1); mask = cv2.bitwise_and(gray, gray, mask=mask)
        maskplot = np.copy(mask)
        cv2.imwrite(self.folder + self.fname.replace(self.extn, "_stg0.1_out" + self.extn), mask)
        thds = [16, 24, 32, 48, 64, 96, 128, 255]
        linked_list = []
        for td in thds:
            masked = np.zeros_like(mask)
            masked[mask <= td] = 255
            masked = cv2.bitwise_and(masked, masked, mask=mask)
            blurs = cv2.blur((255 - masked).astype(np.uint8), (3, 3))
            contours, hierarchy = cv2.findContours(blurs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for ix in range(len(contours)):
                if ix > 0 and hierarchy[0,ix,3] == 0:
                    cimg = np.zeros_like(gray)
                    cv2.drawContours(cimg, contours, int(ix), color=255, thickness=-1)
                    pts = np.where(cimg == 255)
                    intensity = self.intensity_threshold-gray[pts[0], pts[1]].ravel().astype(int)
                    prob = 1-np.mean(1./(1.+np.exp(intensity)))
                    colw = np.array([255,255,255])
                    #print(td, "->", prob)
                    linked_list.append({"prob":prob, "color":255*prob, "contour":contours[int(ix)]})
                    cv2.drawContours(maskplot, contours, int(ix), color=255*prob, thickness=1)
                    self.draw_contour(contours[int(ix)], self.images["prob_masked_image_ol"], col=tuple(colw*prob), line_thick=1)
            cv2.imwrite(self.folder + self.fname.replace(self.extn, "_stg0.2.%03d_out" + self.extn)%td, masked)
        cv2.imwrite(self.folder + self.fname.replace(self.extn, "_stg1.0_out" + self.extn), self.images["prob_masked_image_ol"])
        maskplot = cv2.cvtColor(maskplot, cv2.COLOR_GRAY2RGB)
        maskplot = cv2.applyColorMap(maskplot, cv2.COLORMAP_JET)
        cv2.imwrite(self.folder + self.fname.replace(self.extn, "_stg1.1_out" + self.extn), maskplot)
        self.plot_image_dev(self.images["prob_masked_image_ol"], linked_list)
        self.plot_image_dev(maskplot, linked_list, cmap=mpl.cm.jet, ext="_cb_jet")
        return self
    
    def plot_image_dev(self, image, objs, shape=(15,14), cmap=mpl.cm.gray, ext="_cb_gray"):
        df = pd.DataFrame.from_records(objs)[["prob","color"]]
        fig = plt.figure(figsize=(4, 4), dpi=180)
        ax = plt.subplot2grid(shape, (0,0), rowspan=shape[0]-2, colspan=shape[0]-2)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax = plt.subplot2grid(shape, (2,shape[0]-2), rowspan=9, colspan=1)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        orientation="vertical")
        cb1.set_label("Pr(CH)")
        fig.savefig(self.folder + self.fname.replace(self.extn, ext + self.extn), bbox_inches="tight")
        return
    
    def draw_contour(self, contour, img, gray=False, col=None, line_thick=None):
        if line_thick is None: line_thick = self.draw_param["contur"]["thick"]
        if col is None: col=tuple(self.draw_param["contur"]["color"])
        for _x in range(contour.shape[0]-1):
            if gray: cv2.line(img, (contour[_x,0,0], contour[_x,0,1]), (contour[_x+1,0,0], contour[_x+1,0,1]), (255,255,255), line_thick)
            else: cv2.line(img, (contour[_x,0,0], contour[_x,0,1]), (contour[_x+1,0,0], contour[_x+1,0,1]), col, line_thick)
        if gray: cv2.line(img, (contour[0,0,0], contour[0,0,1]), (contour[-1,0,0], contour[-1,0,1]), (255,255,255), line_thick)
        else: cv2.line(img, (contour[0,0,0], contour[0,0,1]), (contour[-1,0,0], contour[-1,0,1]),  col, line_thick)
        return
    
    def save_files_outputs(self):
        return self

def run_skynet(args, save=True):
    aia = RegisterAIA(args.date, args.wavelength, args.resolution, vmin=10)
    load = Loader(aia.fname, aia.folder, args.date, args, save)
    load.load_model().run_forwarding().save_outputs().estimate_CHB()
    load.calculate_probabilistic_boundaries()
    if save: load.save_files_outputs()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Unsupervised Segmentation")
    parser.add_argument("-dn", "--date", default=dt.datetime(2018,5,30,12), help="Date [2018,5,30,12]", type=prs.parse)
    parser.add_argument("-r", "--resolution", default=1024, help="Resolution of the files [1024]", type=int)
    parser.add_argument("-w", "--wavelength", default=193, help="Wavelength of the files [193]", type=int)
    parser.add_argument("--nChannel", metavar="N", default=50, type=int, help="number of channels")
    parser.add_argument("--maxIter", metavar="T", default=100, type=int, help="number of maximum iterations")
    parser.add_argument("--minLabels", metavar="minL", default=3, type=int, help="minimum number of labels")
    parser.add_argument("--lr", metavar="LR", default=0.3, type=float, help="learning rate")
    parser.add_argument("--nConv", metavar="M", default=2, type=int, help="number of convolutional layers")
    parser.add_argument("--stepsize_sim", metavar="SIM", default=1, type=float, help="step size for similarity loss", required=False)
    parser.add_argument("--stepsize_con", metavar="CON", default=1, type=float, help="step size for continuity loss")
    parser.add_argument("--los", metavar="LOS", default=.1, type=float, help="Final loss value")
    args = parser.parse_args()
    print("\n Parameter list for SkyNet simulation ")
    for k in vars(args).keys():
        print("     " + k + "->" + str(vars(args)[k]))
    run_skynet(args)
