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
    TRAIN = 'ImageSets/Segmentation/train.txt'
    VAL = 'ImageSets/Segmentation/val.txt'
    IMAGE_FOLDER = 'JPEGImages'
    SEGMENTATON_FOLDER = 'SegmentationClass'

    def __init__(self, root, train, num_examples):
        super().__init__()
        # gather all files names of reqeusted data(train/test)
        if train:
            with open(os.path.join(root, VOC2007Dataset.TRAIN)) as f:
                all_files = f.readlines()
        else:
            with open(os.path.join(root, VOC2007Dataset.VAL)) as f:
                all_files = f.readlines()

        if num_examples is not -1:
            files = all_files[:num_examples]
        else:
            files = all_files

        # get dirs
        seg_dir = os.path.join(root, VOC2007Dataset.SEGMENTATON_FOLDER)
        img_dir = os.path.join(root, VOC2007Dataset.IMAGE_FOLDER)
        # set images and segmentations
        self.imgs = [os.path.join(img_dir, f.strip() + '.jpg') for f in files]
        self.segmentations = [os.path.join(seg_dir, f.strip() + '.png') for f in files]

    def __getitem__(self, index):
        example_dict = dict()
        # read images and segmentations
        im = plt.imread(self.imgs[index]) / 255
        gt_raw = 255 * plt.imread(self.segmentations[index])
        gt = np.zeros((1, gt_raw.shape[0], gt_raw.shape[1]))

        # convert mask color to id
        for id, color in enumerate(VOC_LABEL2COLOR):
            mask = (gt_raw == color).all(2)
            gt[:, mask] = id

        example_dict['im'] = numpy2torch(im).float()
        example_dict['gt'] = torch.Tensor(gt).long()
        assert (isinstance(example_dict, dict))
        assert ('im' in example_dict.keys())
        assert ('gt' in example_dict.keys())
        return example_dict

    def __len__(self):
        return len(self.imgs)


def create_loader(dataset, batch_size, shuffle, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))

    # colored_mask = np.zeros_like(np_image)

    r = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.float32)
    g = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.float32)
    b = np.zeros((np_image.shape[0], np_image.shape[1]), dtype=np.float32)
    # convert id to color
    for id, color in enumerate(VOC_LABEL2COLOR):
        mask = (np_label == id).all(2)

        r[mask] = color[0] / 255
        g[mask] = color[1] / 255
        b[mask] = color[2] / 255

    mask_color = np.stack((r, g, b), 2)
    # blend image with color mask
    alpha = 0.7
    colored = mask_color * alpha + np_image * (1 - alpha)

    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    plt.plot()
    plt.title(title)

    plots_done = 0

    for data in loader:
        plots_done += 1
        mask = torch2numpy(data['gt'][0])
        img = torch2numpy(data['im'][0])
        im_masked = voc_label2color(img, mask)

        plt.subplot(grid_height, grid_width, plots_done)
        plt.axis('off')
        plt.imshow(im_masked)

        if plots_done >= grid_height * grid_width:
            break

    plt.show()


def standardize_input(input_tensor):
    mean = torch.Tensor(VOC_STATISTICS['mean']).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    std = torch.Tensor(VOC_STATISTICS['std']).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    # 1. center shift 2. scale/normalize
    normalized = (input_tensor - mean) / std

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized


def run_forward_pass(normalized, model):
    model.eval()
    predict = model(normalized)

    acts = predict['out']
    prediction = torch.max(acts, 1, keepdim=True)[1]

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts


def average_precision(prediction, gt):
    num_labels = prediction.size(3) * prediction.size(2) * prediction.size(1) * prediction.size(0)
    correct_labels = (prediction == gt).sum()
    precision = correct_labels.float() / num_labels
    return precision


def show_inference_examples(loader, model, grid_height, grid_width, title):
    plt.figure()
    plt.title(title)

    plot_done = 0
    for data in loader:
        plot_done += 1
        # load image and run forward pass
        im = standardize_input(data['im'])
        prediction, act = run_forward_pass(im, model)
        # convert 2 numpy
        img = torch2numpy(data['im'][0])
        gt_mask = torch2numpy(data['gt'][0])
        prediction_mask = torch2numpy(prediction[0])
        # mask image with segmentations
        gt_im = voc_label2color(img, gt_mask)
        prediction_im = voc_label2color(img, prediction_mask)
        # plot
        ax = plt.subplot(grid_height, grid_width, plot_done)
        plt.axis('off')
        ax.set_title('avg_prec=%.2f' % average_precision(prediction, data['gt']))
        plt.imshow(np.concatenate((gt_im, prediction_im), 1))

        if plot_done >= grid_height * grid_width:
            break

    plt.show()


def find_unique_example(loader, unique_foreground_label):
    example = []

    for data in loader:
        # get a ground truth
        gt = data['gt']
        unique_sample = torch.unique(gt)
        # has only fore/background Label
        if unique_sample.size(0) == 2:
            # foreground is wanted label
            if unique_foreground_label in unique_sample and 0 in unique_sample:
                example = data
                break

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    # load image and run forward pass
    im = standardize_input(example_dict['im'])
    prediction, act = run_forward_pass(im, model)
    # convert 2 numpy
    img = torch2numpy(example_dict['im'][0])
    gt_mask = torch2numpy(example_dict['gt'][0])
    prediction_mask = torch2numpy(prediction[0])
    # mask image with segmentations
    gt_im = voc_label2color(img, gt_mask)
    prediction_im = voc_label2color(img, prediction_mask)
    # plot
    plt.plot()
    plt.axis('off')
    plt.title('avg_prec=%.2f' % average_precision(prediction, example_dict['gt']))
    plt.imshow(np.concatenate((gt_im, prediction_im), 1))
    plt.show()


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    model.eval()
    # cross entropy function
    cross_entropy = torch.nn.CrossEntropyLoss()
    # get example data
    im = example_dict['im']
    gt = example_dict['gt']
    # normalize image
    im_standardized = standardize_input(im)
    # replace label for target
    target = gt.clone()
    target[gt == src_label] = target_label
    target.squeeze_(1)

    im_standardized.requires_grad = True
    # one step
    def closure():
        opt.zero_grad()
        # get Loss
        prediction, act = run_forward_pass(im_standardized, model)
        loss_f = cross_entropy(act, target)
        loss_f.backward()
        # mask out background
        mask_bg = gt == 0
        mask_bg = torch.cat((mask_bg, mask_bg, mask_bg), 1)
        im_standardized.grad[mask_bg] = 0
        return loss_f

    # opt = optim.SGD([im_standardized], lr=learning_rate)
    opt = optim.LBFGS([im_standardized], lr=learning_rate, max_iter=1)
    for it in range(iterations):
        # step LBFGS
        opt.step(closure)
        # print("iteration: ", it)


    final_prediction, _ = run_forward_pass(im_standardized, model)

    precision = average_precision(final_prediction, example_dict['gt'])

    mean = torch.Tensor(VOC_STATISTICS['mean']).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    std = torch.Tensor(VOC_STATISTICS['std']).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    # convert final
    final_im = (im_standardized.detach() * std) + mean
    final_im = torch.clamp(final_im, 0.0, 1.0)
    final_im = torch2numpy(final_im[0])

    img = torch2numpy(im[0])
    # difference image
    difference = img - final_im
    difference = np.abs(difference)

    final_prediction = voc_label2color(img, torch2numpy(final_prediction[0]))
    # plots
    plt.figure()
    plt.title('avg_prec=%.2f' % precision)

    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.imshow(img)


    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.imshow(final_im)


    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.imshow(difference)


    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.imshow(final_prediction)

    plt.show()


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
    # show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    # show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load Deeplab network
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21)

    # Apply deeplab. Switch to training loader if you want more variety.
    # show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    # cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    # show_unique_example(cat_example, model=model)
    # show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..
    # attack2: convert dog to cat
    dog_example = find_unique_example(valid_loader, unique_foreground_label=12)
    show_unique_example(dog_example, model=model)
    show_attack(dog_example, model, src_label=12, target_label=8, learning_rate=1.0, iterations=5)


if __name__ == '__main__':
    problem2()
