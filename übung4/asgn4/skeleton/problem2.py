from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from matplotlib import pyplot as plt
from skimage import color
from scipy.sparse import csr_matrix

import torch
from torch import optim
from torch.nn import functional as tf
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS
from utils import numpy2torch
from utils import torch2numpy


class VOC2007Dataset(Dataset):
    def __init__(self, root, train, num_examples):
        super().__init__()

    def __getitem__(self, index):
        example_dict = dict()
        assert (isinstance(example_dict, dict))
        assert ('im' in example_dict.keys())
        assert ('gt' in example_dict.keys())
        return example_dict

    def __len__(self):
        return None


def create_loader(dataset, batch_size, shuffle, num_workers):
    loader = []
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))

    colored = []

    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    pass


def standardize_input(input_tensor):
    normalized = []

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized


def run_forward_pass(normalized, model):
    prediction = []
    acts = []

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts


def average_precision(prediction, gt):
    return None


def show_inference_examples(loader, model, grid_height, grid_width, title):
    pass


def find_unique_example(loader, unique_foreground_label):
    example = []

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    pass


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    pass


def problem2():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder
    root = os.environ["VOC2007_HOME"]

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    # you can safely assume batch_size=1 in our tests..
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load Deeplab network
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)

    # Apply deeplab. Switch to training loader if you want more variety.
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..


if __name__ == '__main__':
    problem2()
