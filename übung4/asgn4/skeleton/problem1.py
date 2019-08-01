from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import scipy.cluster as cl
import matplotlib.colors as col
from scipy.sparse import lil_matrix
from sklearn.mixture import GaussianMixture


# you will need to include some libraries depending on your needs
# e.g. scipy.signal
# e.g. scipy.cluster
# e.g. scipy.spatial
# e.g. sklearn.mixture.GaussianMixture
# ..


def edges4connected(height, width):
    """ Construct edges for 4-connected neighborhood MRF. Assume row-major ordering.

      Args:
        height of MRF.
        width of MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """

    # construct a matrix filled with indices
    npixels = height * width
    idx = np.arange(npixels).reshape(height, width)
    # horizontal edges
    hedges = np.hstack((idx[:, :-1].reshape((-1, 1)), idx[:, 1:].reshape((-1, 1))))
    # vertical edges
    vedges = np.hstack((idx[:-1, :].reshape((-1, 1)), idx[1:, :].reshape((-1, 1))))
    # stack
    edges = np.vstack((hedges, vedges))

    # sanity check
    assert (edges.shape[0] == 2 * (height * width) - (height + width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges


def cluster_coarse_color_statistics(smoothed, num_segments):
    """ Performs initial clustering of color statistics of (smoothed) input

      Args:
        smoothed            input image
        num_segments        number of clusters

      Returns:
        initial labels in {0, ..., NUM_LABELS-1} of size MxN
    """
    inp = np.reshape(smoothed, (smoothed.shape[0] * smoothed.shape[1], smoothed.shape[2]))

    _, label = cl.vq.kmeans2(inp, num_segments)

    label = np.reshape(label, smoothed.shape[0:2])
    assert (label.ndim == 2 and np.equal(label.shape, smoothed.shape[0:2]).all())
    assert (label.dtype in [np.int32, np.int64])
    return label


def label2color(im, label):
    """ Returns a color-coding of labels superimposed on input image

      Args:
        im          input image of size MxNx3
        label       pixelwise labels in {0, ..., NUM_LABELS-1} of size MxN

      Returns:
        Color-coded labels
    """
    colored = col.rgb_to_hsv(im);
    for x in range(colored.shape[0]):
        for y in range(colored.shape[1]):
            colored[x, y, 0] = label[x, y] * 0.33
    colored = col.hsv_to_rgb(colored);
    assert (np.equal(colored.shape, im.shape).all())
    assert (im.dtype == colored.dtype)
    return colored


def contrast_weight(im, edges):
    """ Computes the weight vector W for the contrast-sensitive Potts model:
        For each pairwise potential e connecting pixels i with j compute
        w(e) = exp(- beta*||I(i) - I(j)||_2^2), where beta is the mean
        squared distance between the intensity values of neighboring pixels.

      Args:
        im           input image
        edges        edge map

      Returns:
        Contrast sensitive weights for every edge
    """
    im = np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))

    n = edges.shape[0]
    Sum = 0
    for i in range(n):
        Sum += np.linalg.norm(im[edges[i, 0]] - im[edges[i, 1]]) ** 2

    beta = n / Sum

    cweights = np.zeros(edges.shape[0])
    for i in range(edges.shape[0]):
        cweights[i] = np.exp(-beta * np.linalg.norm(im[edges[i, 0]] - im[edges[i, 1]]) ** 2)
    # sanity check
    assert (len(cweights[:]) == edges.shape[0])
    return cweights


def make_pairwise(lmbda, edges, cweights, num_sites):
    """ Make pairwise capacity matrix for contrast-sensitive Potts model """
    # construct pairwise edges
    pairwise = lil_matrix((num_sites, num_sites), dtype=np.float32)
    # use given potts model
    for edge, cv in zip(edges, cweights):
        pairwise[edge[0], edge[1]] = cv * lmbda
    pairwise = pairwise.tocsr()
    assert (isinstance(pairwise, csr_matrix))
    assert (np.equal(pairwise.shape, (num_sites, num_sites)).all())
    return pairwise


def negative_logprob_gmm(im, label, gmm_components, num_segments):
    """ Fits and evaluates Gaussian mixture models on segments

      Args:
        im                  input image
        label               current labels
        gmm_components      number of mixture model components
        num_segments        total number of segments

      Returns:
        nllh negative log probabilities for all pixels
    """
    nllh = []

    lab = np.reshape(label, (-1, 1))
    gmm = GaussianMixture(n_components=gmm_components, covariance_type="full")
    nll = gmm.fit(np.reshape(label, (-1, 1)))
    nllh = nll.score_samples(np.reshape(im, (-1, 1)))
    print(nllh.shape)
    nllh = np.reshape(nllh, (num_segments, label.shape[0] * label.shape[1]))
    print(nllh[:,0:5])
    assert (np.equal(nllh.shape, (num_segments, label.shape[0] * label.shape[1])).all())
    return nllh


def expand_alpha(alpha, im, label, pairwise, gmm_components, num_segments):
    """ Perform single step of alpha-expansion

      Args:
        alpha               current label to expand
        im                  input image
        label               current labels
        pairwise            pairwise capacity matrix
        gmm_components      number of mixture model components
        num_segments        total number of segments

      Returns:
        label mask (with alpha having been expanded)
    """
    updated_label = []

    # you should call negative_logprob_gmm eventually...

    assert (np.equal(updated_label.shape, label.shape).all())
    assert (updated_label.dtype == label.dtype)
    return updated_label


def iterated_graphcuts(im, label0, pairwise, gmm_components, num_segments):
    """ Performs multi-label iterated graph cuts segmentation

      Args:
        im                  input image
        label0              initial labels
        pairwise            pairwise capacity matrix
        gmm_components      number of mixture model components
        num_segments        total number of segments

      Returns:
        final label mask
    """
    label = []

    # you should call expand_alpha eventually...

    assert (label.ndim == 2 and np.equal(label.shape, label0.shape[0:2]).all())
    assert (label.dtype in [np.int32, np.int64])
    return label


def problem1():
    # Find some good settings: Keep it simple!
    num_segments = 3
    gmm_components = 3  # Find a good parameter
    lmbda = 2.0  # Find a good parameter

    # Read input image
    im = plt.imread('elk.png')

    # Cluster color statistics: Yields coarse estimate
    # (This part can be extended to give a better initialization)
    label0 = cluster_coarse_color_statistics(im, num_segments=num_segments)

    # Write initial labeling
    if not os.path.isdir(os.path.join(os.getcwd(), 'bin')):
        os.mkdir(os.path.join(os.getcwd(), 'bin'))
    plt.imsave(os.path.join(os.getcwd(), 'bin', 'init.png'), label2color(im, label0))
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis("off")
    ax1.set_title("Image")
    # will show 5 as black cause autoscaling.
    ax1.imshow(im, cmap='gray')
    ax2.axis("off")
    ax2.set_title("Label_0")
    # will show 0 as black cause autoscaling.
    ax2.imshow(label2color(im, label0), cmap='gray')
    plt.show()
    # Contrast sensitive Potts model
    edges = edges4connected(im.shape[0], im.shape[1])
    cweights = contrast_weight(im, edges)
    pairwise = make_pairwise(lmbda, edges, cweights, im.shape[0] * im.shape[1])
    negative_logprob_gmm(im, label0, gmm_components, num_segments)
    # Perform multi-label graph cuts segmentation
    iterated_graphcuts(im, label0, pairwise, gmm_components, num_segments)


if __name__ == '__main__':
    problem1()
