#!/usr/bin/env python

"""skynet.py: Image segnemtation algoritm for Solar Disk: SkyNet"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

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
        self.m = sunpy.map.Map(Fido.fetch(q))
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
    
    def __init__(self, date, params, save=True):
        self.date = date        
        self.params = params
        self.resolution = params.resolution
        self.wavelength = params.wavelength
        self.save = save
        self.get_file_folder()
        self.use_cuda = torch.cuda.is_available()
        self.im = cv2.imread(self.folder + self.fname)
        self.data = torch.from_numpy( np.array([self.im.transpose( (2, 0, 1) ).astype("float32")/255.]) )
        if self.use_cuda: self.data = self.data.cuda()
        self.data = Variable(self.data)
        return
    
    def get_file_folder(self):
        self.folder = "data/SDO-Database/{:4d}.{:02d}.{:02d}/{:d}/{:04d}/".format(self.date.year, self.date.month, self.date.day,
                                                                             self.resolution, self.wavelength)
        self.fname = "{:4d}_{:02d}_{:02d}_{:d}_{:02d}{:02d}{:02d}_{:04d}.png".format(self.date.year, self.date.month,
                                                                                     self.date.day, self.date.hour,
                                                                                     self.date.minute, self.date.second, 
                                                                                     self.resolution, self.wavelength)
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
        output = self.model(self.data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, self.params.nChannel)
        ignore, target = torch.max(output, 1)
        self.im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([self.label_colours[c % 100] for c in self.im_target])
        im_target_rgb = im_target_rgb.reshape(self.im.shape).astype(np.uint8)
        cv2.imwrite(self.folder + self.fname.replace(".png", "_seg.png"), im_target_rgb)
        return
    
    def estimate_CHB(self):
        return self
    
    def convert_to_binary(self):
        return

def run_skynet(args, save=True):
    aia = RegisterAIA(args.date, args.wavelength, args.resolution, vmin=10)
    load = Loader(aia.fname, aia.folder, args.date, args, save)
    load.load_model().run_forwarding().estimate_CHB()
    if save: load.save_outputs()
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