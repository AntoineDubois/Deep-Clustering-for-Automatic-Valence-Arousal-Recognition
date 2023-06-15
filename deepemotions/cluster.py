import numpy as np 

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as KMeans_scipy
from scipy.spatial.distance import cdist

def normalize_columns(columns):
    """
    Normalize columns of matrix.
    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns
    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns/np.sum(columns, axis=0, keepdims=1)

    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.
    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent
    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe
    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x/np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x**exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x**exponent

    result = normalize_columns(x)

    return result


def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.
    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.
    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.
    Parameters
    ----------
    data : 2d array (N x Q)
        Data to be analyzed. There are N data points.
    centers : 2d array (C x Q)
        Cluster centers. There are C clusters, with Q features.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    Returns
    -------
    dist : 2d array (C x N)
        Euclidean distance from each point, to each cluster center.
    See Also
    --------
    scipy.spatial.distance.cdist
    """
    return cdist(data, centers, metric=metric).T


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans_fct(data, c, m, error, maxiter,
           metric='euclidean',
           init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (c, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    Returns
    -------
    cntr : 2d array, size (c, S)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (c, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (c, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (c, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.
    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.
    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.
    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    u2 = 0
    while (p < maxiter - 1) and (np.linalg.norm(u - u2) > error):
        u2 = u.copy()
        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m, "euclidean")
        jm = np.hstack((jm, Jjm))
        p += 1
    
    if p == maxiter -1:
        print("maximum number of iteration reached")

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


class CMeans:
    def __init__(self, signal, time=None, device=None):
        self.signal = signal
        self.time = time

        if self.signal.ndim == 1:
            self.signal = self.signal.reshape(-1, 1)

    def run(self, nb_classes, seq_len, u_init=None, m=1.5, max_iter=300, tol=1e-4):
        data = [self.signal[i-seq_len:i,:] for i in range(seq_len, self.signal.shape[0])]
        data = np.asarray(data)
        data = data.reshape(data.shape[0], np.prod(data.shape[1:]))

        # Setup u0
        if u_init is not None:
            u_init = u_init[seq_len:]
            u_init = u_init.T
        # Main cmeans loop
        self.centroids, u, u0, d, jm, p, fpc = cmeans_fct(data=data.T, c=nb_classes, m=m, error=tol, maxiter=max_iter,
           metric='euclidean', init=u_init)
        u = u.T
        labels_hat = np.argmax(u, axis=1) +1

        if self.time is None:
            return labels_hat, u
        else:
            return labels_hat, u, self.time[seq_len:]

    def init_centroids(self, data, s_init, nb_classes):
        self.centroids = np.zeros((data.shape[1], nb_classes))
        for k in range(nb_classes):
            mask = s_init == k
            self.centroids[:,k] = np.mean(data[mask], axis=0)



class KMeans:
    def __init__(self, signal, time=None, device=None):
        self.signal = signal
        self.time = time

        if self.signal.ndim == 1:
            self.signal = self.signal.reshape(-1, 1)

    def run(self, nb_classes, seq_len, s_init=None, max_iter=300, tol=1e-4):
        data = [self.signal[i-seq_len:i,:] for i in range(seq_len, self.signal.shape[0])]
        data = np.asarray(data)
        data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        # Setup u0
        if s_init is not None:
            s_init = s_init[seq_len:]
            s_init -= 1
            self.init_centroids(data, s_init, nb_classes)
        else:
            self.centroids = "k-means++"

        # Main cmeans loop
        kmeans = KMeans_scipy(n_clusters=nb_classes, init=self.centroids.T, max_iter=max_iter, tol=tol, n_init=1).fit(data)
        labels_hat = kmeans.labels_ +1

        if self.time is None:
            return labels_hat
        else:
            return labels_hat, self.time[seq_len:]

    def init_centroids(self, data, s_init, nb_classes):
        self.centroids = np.zeros((data.shape[1], nb_classes))
        for k in range(nb_classes):
            mask = s_init == k
            self.centroids[:,k] = np.mean(data[mask], axis=0)

    
    

if __name__ == "__main__":
    N_CLASSES = 1_000
    NB_CLASSES = 3
    N = NB_CLASSES * N_CLASSES
    RESCALLING = True
    SEQ_LEN = 20

    X = np.empty(0)
    label = np.empty(0)
    for i in range(NB_CLASSES):
        X_new = np.random.uniform(size=N_CLASSES) + 4*i
        label_new = np.tile(i +1, reps=N_CLASSES)
        X = np.concatenate((X, X_new))
        label = np.concatenate((label, label_new))

    plt.plot(X)
    plt.show()

    if RESCALLING:
        X = (X -X.min())/(X.max() -X.min())

    plt.plot(X)
    plt.show()

    kmeans = KMeans(signal=X)
    labels_hat = kmeans.run(nb_classes=NB_CLASSES, s_init=label, seq_len=SEQ_LEN,
                                max_iter=300, tol=0) 
    plt.plot(X)
    plt.plot(labels_hat)
    plt.show()

    cmeans = CMeans(signal=X)

    u_init = np.zeros((label.size, NB_CLASSES))
    u_init[np.arange(label.size),label.astype(int)] = 1

    labels_hat, u = cmeans.run(nb_classes=NB_CLASSES, u_init=u_init, seq_len=SEQ_LEN,
                                max_iter=300, tol=0)
    plt.plot(X)
    plt.plot(labels_hat)
    plt.show()






