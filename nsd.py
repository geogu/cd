import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def compute_r_c(O, R, k):
    nn = NearestNeighbors(n_neighbors=k).fit(R)
    dst, idx = nn.kneighbors(O)
    c = len(np.unique(idx))
    r = dst[:, -1]
    gap = np.unique(idx)
    return r, c, R[gap]

def compute_ct(S, r, T):
    m = cdist(S, T)
    msk = np.argwhere(m <= r.reshape(-1, 1))
    gap = np.unique(msk[:, 1])
    return len(gap), T[gap]

def compute_nsd(c, ct):
    if c > ct:
        return st.beta.cdf(0.5, c, ct + 1)
    else:
        return 1 - st.beta.cdf(0.5, c, ct)

def nsd_(X1a, X1b, X2a, X2b, k=1, thres=0.05):
    r0, c0, g1 = compute_r_c(X1a, X1b, k)
    r1, c1, g0 = compute_r_c(X1b, X1a, k)

    ct0, gt1 = compute_ct(X1a, r0, X2b)
    ct1, gt0 = compute_ct(X1b, r1, X2a)

    nsd0 = compute_nsd(c0, ct0)
    nsd1 = compute_nsd(c1, ct1)
    print((nsd0, nsd1))

    return nsd0 < thres or nsd1 < thres, g0, g1, gt0, gt1

def nsd(X1, y1, X2, y2):
    return nsd_(X1[y1], X1[~y1], X2[y2], X2[~y2])[0]

if __name__ == '__main__':
    n = 1000
    rs = np.random.RandomState(0)
    X0 = rs.multivariate_normal(np.zeros(2), np.identity(2), n)
    X1 = rs.multivariate_normal(np.zeros(2) + 2, np.identity(2), n)
    XX0 = rs.multivariate_normal(np.zeros(2), np.identity(2), n)
    XX1 = rs.multivariate_normal(np.zeros(2) + 2.1, np.identity(2), n)

    is_drift, g0, g1, gt0, gt1 = nsd(X0, X1, XX0, XX1)
    print(is_drift)

    plt.scatter(X0[:,0], X0[:,1], alpha=0.5, s=10)
    plt.scatter(X1[:,0], X1[:,1], alpha=0.5, s=10)
    plt.scatter(g0[:,0], g0[:,1], marker='+', s=80)
    plt.scatter(g1[:,0], g1[:,1], marker='+', s=80)
    plt.scatter(gt0[:,0], gt0[:,1], marker='+', s=80)
    plt.scatter(gt1[:,0], gt1[:,1], marker='+', s=80)
    plt.show()