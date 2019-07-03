from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import int8

import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from utils import rgb2gray


def edges4connected(height, width):
    """ Construct edges for 4-connected neighborhood MRF. Assume row-major ordering.

      Args:
        height of MRF.
        width of MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """

    edges = np.ndarray(shape=((2 * (height * width) - (height + width)), 2), dtype=np.int32)
    # add all edges to the left and down... iterate an leaveborders out.
    edgeCounter = 0
    indexCounter = 0
    for h in range(height):
        for w in range(width):
            # to the left
            if w < width - 1:
                edges[indexCounter] = [edgeCounter, edgeCounter + 1]
                indexCounter += 1
            # down
            if h < height - 1:
                edges[indexCounter] = [edgeCounter, edgeCounter + width]
                indexCounter += 1
            edgeCounter += 1
    # print(edges)
    # sanity check
    assert (edges.shape[0] == 2 * (height * width) - (height + width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges


def negative_log_laplacian(x, s):
    """ Elementwise evaluation of a log Laplacian. """

    # nllh for x = i_0 - i_1_w
    def equation_1(x):
        return -np.log(1 / (2 * s)) + (np.fabs(x) / s)

    functionNNL = np.vectorize(equation_1)
    result = functionNNL(x)

    assert (np.equal(result.shape, x.shape).all())
    return result


def negative_stereo_loglikelihood(i0, i1, d, s, invalid_penalty=1000.0):
    """ Elementwise stereo negative log likelihood.

      Args:
        i0, i1                  stereo pair of images.
        d                       given disparities.
        invalid_penalty:        penalty value if disparity yields position outside of valid range.

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    # shift i_1 for d... invalid penalty when out of range of i_1
    nllh = []
    i1Copy = np.copy(i1)
    for m in range(d.shape[0]):
        for n in range(d.shape[1]):
            idx = int(np.round((n - d[m][n])))
            if (0 <= idx) and (idx < i1.shape[1]):
                i1Copy[m][n] = i1[m][idx]
            else:
                i1Copy[m][n] = invalid_penalty

    nllh = negative_log_laplacian(i0 - i1Copy, s)

    assert (np.equal(nllh.shape, d.shape).all())
    assert (nllh.dtype in [np.float32, np.float64])
    return nllh


def alpha_expansion(i0, i1, edges, d0, candidate_disparities, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        i0, i1:                  Given grayscale images.
        edges:                   Given neighboor of MRF.
        d0:                      Initial disparities.
        candidate_disparities:   Set of labels to consider
        lmbda:                   Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes anymore.

      Returns:
        A `nd.array` of type `int32`. Assigned labels (source=0 or target=1) minimizing the costs.


    """
    width = i0.shape[1]

    d = d0.copy()
    changes = 1
    while (changes > 0):
        changes = 0
        print(candidate_disparities)
        # pick candidate disparity
        # for first iteration it makes sense to enforce the spatial relationship in order
        for cand_disparity in candidate_disparities:
            # construct unary endges for picked cand_disparity
            nllh_init = negative_stereo_loglikelihood(i0, i1, d, s)
            nllh_candidate = negative_stereo_loglikelihood(i0, i1, cand_disparity * np.ones(d0.shape), s)
            unary = np.stack((nllh_init.reshape(-1), nllh_candidate.reshape(-1)))
            #construct pairwise edges
            N = d0.shape[0] * d0.shape[1]
            pairwise = lil_matrix((N, N), dtype=np.float32)
            # use given potts model
            for edge in edges:
                if d[int(np.floor(edge[0] / width)), int(edge[0] % width)] == \
                        d[int(np.floor(edge[1] / width)), int(edge[1] % width)]:
                    pairwise[edge[0], edge[1]] = lmbda
            # solve graphcut
            labels = gco.graphcut(unary, pairwise.tocsr())

            labels = labels.reshape(d0.shape)
            #apply labels
            d[0 < labels] = cand_disparity
            print(labels.sum())
            #sum up changes
            changes += labels.sum()
        # for any further iteration permutate candidates
        candidate_disparities = np.random.permutation(candidate_disparities)

    assert (np.equal(d.shape, d0.shape).all())
    assert (d.dtype == d0.dtype)
    return d


def show_stereo(d, gt):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    non_zero = np.where(gt > 0)
    minimum = np.amin(non_zero, axis=1)
    maximum = np.amax(non_zero, axis=1)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis("off")
    ax2.set_title("GT")
    # will show 5 as black cause autoscaling.
    ax1.imshow(gt[minimum[0]:maximum[0], minimum[1]:maximum[1]], cmap='gray')
    ax1.axis("off")
    ax2.set_title("d")
    # will show 0 as black cause autoscaling.
    ax2.imshow(d[minimum[0]:maximum[0], minimum[1]:maximum[1]], cmap='gray')
    plt.show()

    return


def evaluate_stereo(d, gt):
    """Computes percentage of correct labels in the valid region (gt > 0)."""
    # get all greater 0
    non_zero = np.where(gt > 0)
    # find min/max index
    minimum = np.amin(non_zero, axis=1)
    maximum = np.amax(non_zero, axis=1)
    #crop
    gt_crop = gt[minimum[0]:maximum[0], minimum[1]:maximum[1]]
    d_crop = d[minimum[0]:maximum[0], minimum[1]:maximum[1]]
    # find all correct labels
    result = np.where(gt_crop == d_crop)
    # calc percentage
    return result[0].shape[0] / (gt_crop.shape[0] * gt_crop.shape[1])


def problem1():
    # Read stereo images and ground truth disparities
    i0 = rgb2gray(plt.imread('i0.png')).squeeze().astype(np.float32)
    i1 = rgb2gray(plt.imread('i1.png')).squeeze().astype(np.float32)
    gt = (255 * plt.imread('gt.png')).astype(np.int32)

    # Set Potts penalty
    lmbda = 3.0
    s = 10.0 / 255.0

    # Create 4 connected edge neighborhood
    edges = edges4connected(i0.shape[0], i0.shape[1])

    # Candidate search range
    candidate_disparities = np.arange(0, gt.max() + 1)

    # Graph cuts with zero initialization
    zero_init = np.zeros(gt.shape).astype(np.int32)
    estimate_zero_init = alpha_expansion(i0, i1, edges, zero_init, candidate_disparities, s, lmbda)
    show_stereo(estimate_zero_init, gt)
    perc_correct = evaluate_stereo(estimate_zero_init, gt)
    print("Correct labels (zero init): %3.2f%%" % (perc_correct*100))

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=gt.max() + 1, size=i0.shape)
    estimate_random_init = alpha_expansion(i0, i1, edges, random_init, candidate_disparities, s, lmbda)
    show_stereo(estimate_random_init, gt)
    perc_correct = evaluate_stereo(estimate_random_init, gt)
    print("Correct labels (random init): %3.2f%%" % (perc_correct*100))


if __name__ == '__main__':
    problem1()
