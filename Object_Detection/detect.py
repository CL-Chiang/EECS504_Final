from __future__ import division

import os
import sys
sys.path.insert(0, "./YOLOv3/")


from models import *
from utils.utils import *
from utils.datasets import *

import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2

class Detection_Interface():
    """docstring for Detection"""
    def __init__(self, models=None):
        self.models = models

    def detect(self, img):
        return [model.detect(img) for model in self.models]

class YOLO_V3(object):
    """docstring for YOLO_V3"""
    def __init__(self, opt):
        super(YOLO_V3, self).__init__()
        self.opt = opt
        print(self.opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set up model
        self.model = Darknet(self.opt.model_def, img_size=self.opt.img_size).to(device)
        if self.opt.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.opt.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.opt.weights_path, map_location=device))
    


    def detect(self,img):

        self.model.eval()  # Set in evaluation mode

        classes = load_classes(self.opt.class_path)  # Extracts class labels from file

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Bounding-box colors

        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        draw_img = img.copy()

        img = preprocess(img, self.opt.img_size)
        img = Variable(img.type(Tensor))
        img = img.unsqueeze(0)
        with torch.no_grad():
            detections = self.model(img)
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)
        # print(detections, "?")
        if detections[0] is None:
            return
        detections = rescale_boxes(detections[0], self.opt.img_size, draw_img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")


        return detections