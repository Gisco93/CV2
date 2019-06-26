import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as tf
import torch.optim as optim

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image


def numpy2torch(array):
    """ Converts 3D numpy HWC ndarray to 3D PyTorch CHW tensor."""
    assert (array.ndim == 3)
    result = torch.from_numpy(np.swapaxes(np.swapaxes(array, 0, 1), 0, 2))
    return result


def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    result = np.swapaxes(np.swapaxes(tensor.numpy(), 0, 2), 0, 1)
    return result


def load_data(im1_filename, im2_filename, flo_filename):
    """ Loads images and flow ground truth. Returns 4D tensors."""
    img1 = rgb2gray(read_image(im1_filename))
    img2 = rgb2gray(read_image(im2_filename))
    flo = read_flo(flo_filename)
    tensor1 = numpy2torch(img1).unsqueeze_(0)
    tensor2 = numpy2torch(img2).unsqueeze_(0)
    flow_gt = numpy2torch(flo).unsqueeze_(0)
    # print(tensor1.shape)
    # print(tensor2.shape)
    # print(flow_gt.shape)
    warp_image(tensor1, flow_gt)
    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """
    Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.
    """
    assert (flow.dim() == 4 and flow_gt.dim() == 4)
    assert (flow.size(1) == 2 and flow_gt.size(1) == 2)
    epe = torch.norm(torch.where(flow_gt > 1e9, flow - flow_gt, torch.zeros_like(flow_gt)), p=2, dim=1)
    print("max EPE : {}".format(torch.max(epe)))
    print("min EPE : {}".format(torch.min(epe)))
    aepe = epe.mean()

    return aepe


def warp_image(im, flow):
    """ Warps given image according to the given optical flow."""
    assert (im.dim() == 4 and flow.dim() == 4)
    assert (im.size(1) in [1, 3] and flow.size(1) == 2)
    # TODO check right dimensional order
    print("Image shape : {}".format(str(im.shape)))
    dif = torch.div(flow,flow.shape[1])
    flo = flow.transpose(1, 3).transpose(1, 2)

    print("Flow shape : {}".format(str(flow.shape)))
    print("Flo  shape : {}".format(str(flo.shape)))
    warped = torch.nn.functional.grid_sample(im, flo)
    print("warped shape : {}".format(str(warped.shape)))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(torch2numpy(im.squeeze(0))[:, :, 0], cmap='gray')
    ax2.imshow(torch2numpy(flow.squeeze(0))[:, :, 0], cmap='gray')
    ax3.imshow(torch2numpy(warped.squeeze(0))[:, :, 0], cmap='gray')
    plt.show()
    return warped


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) in [1, 3] and im2.size(1) in [1, 3] and flow_gt.size(1) == 2)
    im2_w = warp_image(im2, flow_gt)
    im_diff = torch.zeros_like(im1)
    im_diff = torch.sub(im1, im2_w)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(torch2numpy(im2.squeeze(0))[:, :, 0], cmap='gray')
    ax2.imshow(torch2numpy(flow_gt.squeeze(0))[:, :, 0], cmap='gray')
    ax3.imshow(torch2numpy(flow_gt.squeeze(0))[:, :, 1], cmap='gray')
    plt.show()
    return


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow.size(1) == 2)

    energy = []

    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow]
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)

    return


def estimate_flow_LBFGS(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimates flow using HS with LBFGS.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)

    return


def estimate_flow_coarse_to_fine(im1, im2, flow_gt, lambda_hs, learning_rate,
                                 num_iter, num_level):
    """
    Estimates flow using HS with LBFGS in a coarse-to-fine scheme.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)

    return


def problem2():
    # Loading data
    im1, im2, flow_gt = load_data("frame10.png", "frame11.png", "flow10.flo")

    # Parameters
    lambda_hs = 0.0015
    num_iter = 400

    print(im1.size(1))
    print(im2.size(1))
    print(flow_gt.size(1))
    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)

    # Gradient descent
    learning_rate = 20
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)

    # LBFGS
    learning_rate = 1
    estimate_flow_LBFGS(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)

    # Coarse to fine
    learning_rate = 1
    num_level = 4
    estimate_flow_coarse_to_fine(
        im1, im2, flow_gt, lambda_hs, learning_rate, num_iter, num_level)


if __name__ == "__main__":
    problem2()
