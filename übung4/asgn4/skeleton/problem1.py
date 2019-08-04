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
    # use kmeans for initial Segmentation
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
    colored = col.rgb_to_hsv(im)
    for x in range(colored.shape[0]):
        for y in range(colored.shape[1]):
            colored[x, y, 0] = label[x, y] * 0.33
    colored = col.hsv_to_rgb(colored)
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
    # compute beta as given in formular
    n = edges.shape[0]
    Sum = 0
    for i in range(n):
        Sum += np.linalg.norm(im[edges[i, 0]] - im[edges[i, 1]]) ** 2

    beta = n / Sum
    # compute weights as in formular 3
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
    for edge, cw in zip(edges, cweights):
        pairwise[edge[0], edge[1]] = cw * lmbda
    #convert to csr
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
    nllh = None
    # This approach works better in the spatial domain
    for seg in range(num_segments):
        label_seg = np.array([x == seg for x in label])
        spatial_seg = []
        spatial_im = []
        # transform Segmentation and Image into spatial domain
        for x in range(label_seg.shape[0]):
            for y in range(label_seg.shape[1]):
                if label_seg[x, y] == 1:
                    spatial_seg.append(np.array([x, y]))
                spatial_im.append(np.array([x, y]))

        spatial_seg = np.array(spatial_seg)
        spatial_im = np.array(spatial_im)
        gmm = GaussianMixture(n_components=gmm_components, covariance_type="full")
        # score segment if not empty
        if not (len(spatial_seg) == 0):
            # fit a segment
            gmm.fit(spatial_seg)
            # compute log Likelihood
            nllh_seg = gmm.score_samples(spatial_im).reshape(-1, 1)
        else:
            nllh_seg = np.zeros((spatial_im.shape[0], 1))
        # stack segments
        if nllh is None:
            nllh = nllh_seg
        else:
            nllh = np.hstack([nllh, nllh_seg])

    # it would make more sense to keep the image in the color domain but the nllh result looks anything but great
    # This gives a result which seems not very smooth as a Model of Gaussian should be:
    # this results in weird Label assignments(merging Labels) for the graphcut

    # for seg in range(num_segments):
    #     label_seg = np.array([x == seg for x in label])
    # 
    #     im_seg = np.zeros_like(im)
    #     im_seg[0 < label_seg, 0] = im[0 < label_seg, 0]
    #     im_seg[0 < label_seg, 1] = im[0 < label_seg, 1]
    #     im_seg[0 < label_seg, 2] = im[0 < label_seg, 2]
    # 
    #     gmm = GaussianMixture(n_components=gmm_components, covariance_type="full")
    # 
    #     gmm.fit(im.reshape(im.shape[0] * im.shape[1], im.shape[2]))
    # 
    #     nllh_seg = - gmm.score_samples(im_seg.reshape(im.shape[0] * im.shape[1], im.shape[2])).reshape(-1, 1)
    #     if nllh is None:
    #         nllh = nllh_seg
    #     else:
    #         nllh = np.hstack([nllh, nllh_seg])

    nllh = nllh.T

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
    updated_label = label.copy()
    # compute unary connections for label image and alpha image
    nllh_init = negative_logprob_gmm(im, label, gmm_components, num_segments)
    nllh_candidate = negative_logprob_gmm(im, alpha * np.ones(label.shape), gmm_components, num_segments)

    for seg in range(num_segments):
        # construct data term
        unary = np.stack((nllh_init[seg, :].reshape(-1), nllh_candidate[seg, :].reshape(-1)))
        # use graphcut
        labels_update = gco.graphcut(unary, pairwise)
        labels_update = labels_update.reshape(label.shape)
        # update to alpha
        updated_label[labels_update == 1] = alpha

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
    label = label0
    old_label = label0 + 11
    # you should call expand_alpha eventually... no fuck off ;P
    c = 0
    # stop when almost no labels change... results are this way good enough and it doesn`t take forever
    while (np.fabs(np.sum(old_label - label)) >= 10.0):
        old_label = label
        c += 1
        for alpha in range(num_segments):
            label = expand_alpha(alpha, im, label, pairwise, gmm_components, num_segments)
        # image saving used:
        if c % 4 == 1:
            print("intermediate")
            plt.imsave(os.path.join(os.getcwd(), 'bin', 'intermediate{}.png'.format(c)), label2color(im, label))
        print("iteration: ", c)
        print(np.sum(old_label - label))
    print("final")
    plt.imsave(os.path.join(os.getcwd(), 'bin', 'final.png'), label2color(im, label))

    assert (label.ndim == 2 and np.equal(label.shape, label0.shape[0:2]).all())
    assert (label.dtype in [np.int32, np.int64])
    return label


def problem1():
    # Find some good settings: Keep it simple!
    num_segments = 3
    gmm_components = 5  # Find a good parameter
    lmbda = 100 # Find a good parameter

    # Read input image
    im = plt.imread('elk.png')

    # Cluster color statistics: Yields coarse estimate
    # (This part can be extended to give a better initialization)
    label0 = cluster_coarse_color_statistics(im, num_segments=num_segments)

    # Write initial labeling
    if not os.path.isdir(os.path.join(os.getcwd(), 'bin')):
        os.mkdir(os.path.join(os.getcwd(), 'bin'))
    plt.imsave(os.path.join(os.getcwd(), 'bin', 'init.png'), label2color(im, label0))

    # Contrast sensitive Potts model
    edges = edges4connected(im.shape[0], im.shape[1])
    cweights = contrast_weight(im, edges)
    pairwise = make_pairwise(lmbda, edges, cweights, im.shape[0] * im.shape[1])

    # Perform multi-label graph cuts segmentation
    iterated_graphcuts(im, label0, pairwise, gmm_components, num_segments)


if __name__ == '__main__':
    problem1()
