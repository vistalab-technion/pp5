import numpy as np
from numpy.random import permutation
from scipy.spatial.distance import pdist, squareform


def tw_test(X, Y, k=1000):
    """
    Calculates the T2 Welch statistic based on distances
    :param X: (N, nx) array containing sample X
    :param Y: (N, ny) array containing sample Y
    :param k: number of permutation for significance evaluation
    :return: t2 statistic, significance
    """

    nx = X.shape[1]  # sample sizes
    ny = Y.shape[1]
    Z = np.hstack((X, Y))  # pooled vectors
    D = 0.5 * squareform(pdist(Z.T)) ** 2  # squared distances

    # T statistic
    def T2(D):
        return (
            (nx + ny)
            / nx
            / ny
            * (
                np.sum(D) / (nx + ny)
                - np.sum(D[0:nx, 0:nx]) / nx
                - np.sum(D[nx:, nx:]) / ny
            )
            / (
                np.sum(D[0:nx, 0:nx]) / (nx ** 2) / (nx - 1)
                + np.sum(D[nx:, nx:]) / (ny ** 2) / (ny - 1)
            )
        )

    # Estimate significance using permutation testing
    t2 = T2(D)
    t2_perm = []
    for i in range(k):
        idx = permutation(nx + ny)
        t2_perm.append(T2(D[idx, :][:, idx]))
    p = np.mean([t2 <= t2_ for t2_ in t2_perm])

    return t2, p
