# fptools: First-passage time calculations
#    Copyright (C) 2021 Garrett Smith
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Tested with CPython 3.8.5 on macOS 11.2.3

import numpy as np
from scipy.linalg import expm
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


def etd(t, T, A, p0):
    """Exit time distribution. Takes a time point, a transient matrix,
    an absorbing matrix, and an initial probability distribution over
    states. Returns the probability density of reaching any absorbing
    state at time t.
    """
    return A.dot(expm(T*t).dot(p0)).sum()


def meanvar(T, p0):
    """Calculates the mean and variance of the exit time distribution.
    All that's needed is the transient matrix T and an initial condition p0.
    """
    mmts = []
    for m in [1, 2]:
        mmts.append(sum((-1)**m * np.math.factorial(m) * (np.linalg.matrix_power(T, -m)).dot(p0)))
    mmts[-1] = mmts[-1] - mmts[0]**2
    return mmts


def splittingprobabilities(T, A, p0):
    """Calculates the splitting probabilities of different
    absorbing states given the transient matrix T, the absorbing
    matrix A, and the initial conditions p0.
    """
    return -A.dot(np.linalg.inv(T).dot(p0))


def cfptd(t, T, A, p0):
    """Returns a vector of the conditional first-passage time
    densities at time t given the transient matrix T, the
    absorbing matrix A, and initial conditions p0.
    """
    return A.dot(expm(T*t).dot(p0)) / splittingprobabilities(T, A, p0)


def cmfpt(T, A, p0):
    """Returns the conditional mean first-passage times given
    the transient matrix T, the absorbing matrix A, and the
    initial conditions p0.
    """
    return A.dot(np.linalg.matrix_power(T, -2).dot(p0)) / splittingprobabilities(T, A, p0)


def index_to_name(n2i):
    """Takes a name-to-index dict and returns a dict with state names as keys
    and indices as values
    """
    return {v: k for k, v in n2i.items()}
    

def make_sys(transitions, scale_rates=True):
    """Takes a string of the form 'old, new, rate' and returns a dict with the
    state names and their index in the W, the full transition matrix W, the
    transient sub-matrix T, and the absorbing matrix A. The string should have
    one transition per line. Scaling rates by multiplying them by the number of
    total states is (system size) is toggled with scale_rates.
    """
    statenames = []
    transdict = {}
    rows = []
    cols = []
    rates = []
    idx = dict()
    ct = 0
    for line in transitions.splitlines():
        old, new, rate = line.split(", ")
        if old not in statenames:
            statenames.append(old)
            idx[old] = ct
            ct += 1
        if new not in statenames:
            statenames.append(new)
            idx[new] = ct
            ct += 1
        rows.append(idx[new])
        cols.append(idx[old])
        rates.append(rate)
    # Sparse matrix for convenient construction, but converted to dense
    W = coo_matrix((rates, (rows, cols)), shape=(len(statenames), len(statenames)),
                     dtype=np.float64).toarray() + 0
    if scale_rates:
        W *= len(statenames)
    np.fill_diagonal(W, -W.sum(axis=0))
    # Getting transient and absorbing submatrices
    idxT, idxA, transdims, absdims = _get_dims(W, idx)
    T = W.copy()[np.ix_(transdims, transdims)]
    A = W.copy()[np.ix_(absdims, transdims)]
    # Checks:
    assert absdims, "No absorbing states given. Check the transitions you provided."
    return idx, idxT, idxA, W, T, A, transdims, absdims


def _get_dims(W, idx=None):
    """Returns the transient and absorbing dimensions of the transition rate
    matrix W. Also returns dicts relating dimension indices and state names. If
    idx is not None, it should be a dict of state name: index pairs for the W
    matrix.
    """
    transdims = []
    absdims = []
    absctr = 0
    transctr = 0
    idxT = dict()
    idxA = dict()
    if idx:
        i2n = index_to_name(idx)
    for i in range(W.shape[0]):
        if W[i,i] == 0:
            absdims.append(i)
            if idx:
                idxA[i2n[i]] = absctr
            absctr += 1
        else:
            transdims.append(i)
            if idx:
                idxT[i2n[i]] = transctr
            transctr += 1
    return idxT, idxA, transdims, absdims


def rescale_matrices(W, tau):
    """Rescale the transition rate matrices by a constant tau. Returns the
    updated W, T, and A matrices.
    """
    V = W.copy()
    np.fill_diagonal(V, 0)
    V *= tau
    np.fill_diagonal(V, -V.sum(axis=0))
    _, _, transdims, absdims = _get_dims(V)
    return V, V[np.ix_(transdims, transdims)], V[np.ix_(absdims, transdims)]


if __name__ == "__main__":
    st = """nostr, corr, 3
corr, nostr, 0.3
corr, abs, 1
nostr, gp, 2
gp, nostr, 0.5"""
    print('Transitions: {}'.format(st))
    idxW, idxT, idxA, W, T, A, transdims, absdims = make_sys(st)
    #p0 = np.array([1., 0, 0])
    p0 = np.zeros(T.shape[0])
    p0[idxT['gp']] = 1
    mn, vr = meanvar(T, p0)
    print('Analytical mean and variance: {}, {}'.format(*map(lambda x:
        np.round(x, 3), [mn, vr])))



