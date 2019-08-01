from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

VOC_STATISTICS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

VOC_LABEL2COLOR = [
    (0, 0, 0),          # 0=background
    (128, 0, 0),        # 1=aeroplane
    (0, 128, 0),        # 2=bicycle
    (128, 128, 0),      # 3=bird
    (0, 0, 128),        # 4=boat
    (128, 0, 128),      # 5=bottle
    (0, 128, 128),      # 6=bus
    (128, 128, 128),    # 7=car
    (64, 0, 0),         # 8=cat
    (192, 0, 0),        # 9=chair
    (64, 128, 0),       # 10=cow
    (192, 128, 0),      # 11=dining table
    (64, 0, 128),       # 12=dog
    (192, 0, 128),      # 13=horse
    (64, 128, 128),     # 14=motorbike
    (192, 128, 128),    # 15=person
    (0, 64, 0),         # 16=potted plant
    (128, 64, 0),       # 17=sheep
    (0, 192, 0),        # 18=sofa
    (128, 192, 0),      # 19=train
    (0, 64, 128)]       # 20=tv/monitor

VOC_LABEL2TXT = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"]


def numpy2torch(array):
    """ Converts 3D numpy HWC ndarray to 3D PyTorch CHW tensor."""
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    elif array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array)


def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    return np.transpose(tensor.numpy(), (1, 2, 0))
