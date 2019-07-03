import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
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
    # numpy array to 3D Tensor
    result = torch.from_numpy(np.swapaxes(np.swapaxes(array, 0, 1), 0, 2))
    return result


def torch2numpy(tensor):
    """ Convert 3D PyTorch CHW tensor to 3D numpy HWC ndarray."""
    assert (tensor.dim() == 3)
    # 3DTensor to numpy array
    result = np.swapaxes(np.swapaxes(tensor.numpy(), 0, 2), 0, 1)
    return result


def load_data(im1_filename, im2_filename, flo_filename):
    """ Loads images and flow ground truth. Returns 4D tensors."""
    # load images as numpy array
    img1 = rgb2gray(read_image(im1_filename))
    img2 = rgb2gray(read_image(im2_filename))
    flo = read_flo(flo_filename)
    # convert to torch 4D tensor
    tensor1 = numpy2torch(img1).unsqueeze_(0)
    tensor2 = numpy2torch(img2).unsqueeze_(0)
    flow_gt = numpy2torch(flo).unsqueeze_(0)
    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """
    Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.
    """
    assert (flow.dim() == 4 and flow_gt.dim() == 4)
    assert (flow.size(1) == 2 and flow_gt.size(1) == 2)
    # EPE only where flow_gt < 1e9
    epe = torch.norm(torch.where(flow_gt < 1e9, flow - flow_gt, torch.zeros_like(flow_gt)), p=2, dim=1)
    # mean of EPE
    aepe = epe.mean()
    return aepe


def warp_image(im, flow):
    """ Warps given image according to the given optical flow."""
    assert (im.dim() == 4 and flow.dim() == 4)
    assert (im.size(1) in [1, 3] and flow.size(1) == 2)
    W, H = im.shape[2:4]
    # norm flow to -1 to 1
    scale = torch.tensor([float(W), float(H)])
    scale.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
    flow_norm = torch.div(flow, scale)
    # adjust shape to grid shape needed by gird sample
    flo = flow_norm.transpose(1, 3).transpose(1, 2)
    # as top left pixel is -1 -1 and bottom right is 1 1 construct 1 to mapping grid
    v_grid = torch.linspace(-1, 1, W).repeat(1, 1, H).view(H, W)
    u_grid = torch.linspace(-1, 1, H).repeat(1, 1, W).view(W, H).transpose(0, 1)
    grid = torch.stack([u_grid, v_grid]).unsqueeze_(0).transpose(1, 3)
    # apply to mapping
    grid = grid + flo
    # warp image using grid sample and bilinear filtering
    warped = torch.nn.functional.grid_sample(im, grid, 'bilinear')
    return warped


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) in [1, 3] and im2.size(1) in [1, 3] and flow_gt.size(1) == 2)
    # warp image 2
    im2_w = warp_image(im2, flow_gt)
    # diff 1 - 2_warped
    im_diff = torch.sub(im1, im2_w)
    # plot
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis("off")
    ax1.set_title("Image 1")
    ax1.imshow(torch2numpy(im1.squeeze(0))[:, :, 0], cmap='gray')
    ax2.axis("off")
    ax2.set_title("Image 2 warped")
    ax2.imshow(torch2numpy(im2_w.squeeze(0))[:, :, 0], cmap='gray')
    ax3.axis("off")
    ax3.set_title("diff")
    ax3.imshow(torch2numpy(im_diff.squeeze(0))[:, :, 0], cmap='gray')
    plt.show()
    return


def plot_flow(flow, flow_gt):
    # plot flow
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis("off")
    ax1.set_title("Flow GT")
    ax1.imshow(flow2rgb(torch2numpy(flow_gt.squeeze(0))))
    ax2.axis("off")
    ax2.set_title("Flow")
    ax2.imshow(flow2rgb(torch2numpy(flow.squeeze(0))))
    plt.show()


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function."""
    assert (im1.dim() == 4 and im2.dim() == 4 and flow.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow.size(1) == 2)
    # warp image
    im2_w = warp_image(im2, flow)
    # construct integral term
    energy = torch.pow(im2_w - im1, 2)
    energy = energy.sum()
    # forward differences
    delta_u = torch.pow(torch.norm(flow[:, :, :, :-1] - flow[:, :, :, 1:]), 2)
    delta_v = torch.pow(torch.norm(flow[:, :, :-1, :] - flow[:, :, 1:, :]), 2)
    # sum the diffs
    energy += lambda_hs * (delta_u + delta_v)

    return energy.sum()


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow]
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)
    # generate some random input
    flow = torch.rand_like(flow_gt) * 10 - 5
    flow.requires_grad = True

    for iteration in range(num_iter):
        #construct loss
        loss = energy_hs(im1, im2, flow, lambda_hs)
        # get gradient
        grad = torch.autograd.grad(loss, flow)[0]
        # apply gradient descent
        flow = flow - learning_rate * grad
        # display AEPE and gradient norm
        print("%03d: %.2f\t %.2f" % (iteration, evaluate_flow(flow, flow_gt), grad.norm()))
    result = flow.detach()
    # visualize flow
    plot_flow(result, flow_gt)
    return result


def estimate_flow_LBFGS(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter, init_flow=None):
    """
    Estimates flow using HS with LBFGS.
    Displays average endpoint error.
    Visualizes flow field.

    Returns estimated flow
    """
    assert (im1.dim() == 4 and im2.dim() == 4 and flow_gt.dim() == 4)
    assert (im1.size(1) == 1 and im2.size(1) == 1 and flow_gt.size(1) == 2)
    # if no init flow given construct random flow
    # manly used for coarse to fine
    flow = init_flow
    if init_flow is None:
        flow = torch.rand_like(flow_gt) * 2 - 1
    flow.requires_grad = True

    optimizer = torch.optim.LBFGS([flow], lr=learning_rate)
    def closure():
        optimizer.zero_grad()
        loss_f = energy_hs(im1, im2, flow, lambda_hs)
        loss_f.backward()
        return loss_f

    for iteration in range(num_iter):
        # make LBFGS step
        optimizer.step(closure)
        # display aepe
        print("%03d: %.2f" % (iteration, evaluate_flow(flow, flow_gt)))

    result = flow.detach()
    # visualize flow
    plot_flow(result, flow_gt)
    return result


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
    flow_sampled = torch.rand_like(flow_gt) * 2 - 1
    print(1 / (2 ** (num_level - 1)))
    for level in reversed(range(num_level)):
        print(1 / (2 ** level))
        im1_sampled = downsample(im1, 1 / (2 ** level), mode='bilinear')
        im2_sampled = downsample(im2, 1 / (2 ** level), mode='bilinear')
        # bilinear has its issues with the 1e9 terms there we use nearest interpolation here
        flow_gt_sampled = downsample(flow_gt, 1 / (2 ** level), mode='nearest')
        flow_sampled = downsample(flow_sampled, size=im1_sampled.size()[2:4], mode='bilinear')
        flow_sampled *= 2
        flow_sampled = estimate_flow_LBFGS(im1_sampled, im2_sampled, flow_gt_sampled, lambda_hs, learning_rate,
                                           num_iter,
                                           init_flow=flow_sampled)
    # visualize flow
    plot_flow(flow_sampled, flow_gt_sampled)

    return flow_sampled


def downsample(im, scale_factor=None, size=None, mode='bilinear'):
    if scale_factor is not None:
        downsample = tf.interpolate(im, scale_factor=scale_factor, mode=mode)
    else:
        downsample = tf.interpolate(im, size=size, mode=mode)
    return downsample


def problem2():

    # Loading data
    im1, im2, flow_gt = load_data("frame10.png", "frame11.png", "flow10.flo")

    # Parameters
    lambda_hs = 0.0015
    num_iter = 400

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
