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
    return A.dot(expm(T*t).dot(p0)) / splpr(T, A, p0)


def cmfpt(T, A, p0):
    """Returns the conditional mean first-passage times given
    the transient matrix T, the absorbing matrix A, and the
    initial conditions p0.
    """
    return A.dot(np.linalg.matrix_power(T, -2).dot(p0)) / splpr(T, A, p0)


def name_to_index(lst):
    """Returns a dict with state names as keys and indices as values

    """
    return {k: v for v, k in enumerate(set(lst))}
    

def index_to_name(n2i):
    """Takes a name-to-index dict and returns a dict with state names as keys
    and indices as values
    """
    return {v: k for k, v in n2i.items()}
    

def make_sys(transitions):
    """Takes a string of the form 'old, new, rate' and returns a dict with the
    state names and their index in the W, the full transition matrix W, the
    transient sub-matrix T, and the absorbing matrix A.
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
    np.fill_diagonal(W, -W.sum(axis=0))
    # Getting transient and absorbing submatrices
    transdims = []
    absdims = []
    absctr = 0
    transctr = 0
    idxT = dict()
    idxA = dict()
    i2n = index_to_name(idx)
    for i in range(W.shape[0]):
        if W[i,i] == 0:
            absdims.append(i)
            idxA[i2n[i]] = absctr
            absctr += 1
        else:
            transdims.append(i)
            idxT[i2n[i]] = transctr
            transctr += 1
    T = W.copy()[np.ix_(transdims, transdims)]
    A = W.copy()[np.ix_(absdims, transdims)]
    # Checks:
    assert absdims, "No absorbing states given. Check the transitions you provided."
    return idx, idxT, idxA, W, T, A, transdims, absdims


def gillespie_until_abs(W, initidx=0):
    probmat = W.copy()
    np.fill_diagonal(probmat, 0.0)
    #probmat /= probmat.sum(axis=0)
    sums = probmat.sum(axis=0)
    probmat = np.divide(probmat, sums, out=np.zeros_like(probmat), where=sums!=0)
    currstate = initidx
    time = 0.0
    # Generating the timestep
    while W[currstate, currstate] != 0:
        time += np.random.exponential(1/-W[currstate, currstate])
        currstate = np.random.choice(range(probmat.shape[0]),
                p=probmat[:,currstate])
    return time


def rep_gillespie(W, initidx=0, n=100):
    fpts = np.zeros(n)
    print('Running {} stochastic simulations until absorbtion'.format(n))
    for i in tqdm(range(n)):
        fpts[i] = gillespie_until_abs(W, initidx)
    return fpts


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

    nsim = 1000
    fpts = rep_gillespie(W, initidx=idxW['gp'], n=nsim)
    print('From stochastic simulations\nMean: {} (SE = {}), Variance: {}'.format(*map(lambda x: np.round(x,
        3), [np.mean(fpts), np.std(fpts)/np.sqrt(nsim), np.var(fpts)])))
    tvec = np.linspace(0, 30, 300)
    plt.hist(fpts, bins=30, density=True, label='FPT (stochastic simulations)')
    plt.plot(tvec, [etd(t, T, A, p0) for t in tvec], label='Analytical FPTD')
    plt.legend()
    plt.show()


