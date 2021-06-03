# Tested with CPython 3.8.5 on macOS 11.2.3

from fp import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ssa_until_abs(W, initidx=0):
    """Run the stochastic simulation algorithm (Gillespie, 1977) until reaching
    an absorbing state. Takes the full transition matrix W and the index of the
    initial state and returns the time of absorption and the absorbing state.
    """
    probmat = W.copy()
    np.fill_diagonal(probmat, 0.0)
    sums = probmat.sum(axis=0)
    probmat = np.divide(probmat, sums, out=np.zeros_like(probmat), where=sums!=0)
    currstate = initidx
    time = 0.0
    # Generating the timestep
    while W[currstate, currstate] != 0:
        time += np.random.exponential(-1/W[currstate, currstate])
        currstate = np.random.choice(range(probmat.shape[0]),
                p=probmat[:,currstate])
    return time, currstate


def rep_ssa(W, initidx=0, n=100):
    """Repeats stochastic runs until absorbtion. Returns a pandas data frame
    with colums for the first-passage/exit time and the absorbing state
    """
    fpts = []
    print('Running {} stochastic simulations until absorbtion'.format(n))
    for i in tqdm(range(n)):
        fpts.append(ssa_until_abs(W, initidx))
    return pd.DataFrame(fpts, columns=['fpt', 'absstate'])


if __name__ == "__main__":
    W = np.array([-2, 1, 1, 0, 0,
                  1, -2., 0, 0, 0,
                  1, 0, -2, 0, 0,
                  0, 1, 0, 0, 0,
                  0, 0, 1, 0, 0]).reshape(5,5)
    T = W[:3, :3]
    A = W[3:, :3]
    init = 0
    p0 = np.zeros(3)
    p0[init] = 1.  # Init. bias toward first abs. st.

    nsim = 1000
    print('Exit times:')
    ets = rep_ssa(W, initidx=init, n=nsim)
    analytical = np.round(meanvar(T, p0), 3)
    print('Analytical results:\n\tMean: {}, Variance: {}'.format(*analytical))
    print('Stochastic simulations\n\tMean: {} (SE = {}), Variance: {}'.format(*map(lambda x: np.round(x,
        3), [ets['fpt'].mean(), ets['fpt'].std()/np.sqrt(ets.shape[0]),
            ets['fpt'].var()])))
    tvec = np.linspace(0, 30, 300)
    ets['fpt'].hist(bins=100, density=True, label='Exit times (stochastic simulations)')
    plt.plot(tvec, [etd(t, T, A, p0) for t in tvec], label='Analytical ETD')
    plt.grid(b=None)
    plt.legend()
    plt.show()

    print('\n\nFirst-passage times:')
    analytical2 = np.round(cmfpt(T, A, p0), 3)
    ansplpr = splittingprobabilities(T, A, p0)
    print('Analytical results:\n\tMean 3: {}, Mean 4: {}'.format(*analytical2))
    print('\tSplitting probabilities: {}'.format(np.round(ansplpr, 3)))
    print('Stochastic simulations:\n\tMean 3: {}, Mean 4: {}'.format(
        *ets.groupby('absstate').mean().round(3).to_numpy().flatten()))
    empspl = (ets.groupby('absstate').count() / ets.shape[0]).to_numpy().flatten().round(3)
    print('\tSplitting probabilities:{}'.format(empspl))
    tvec = np.linspace(0, 15, 200)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ets.hist(ax=ax, column='fpt', by='absstate', bins=50, density=True,
            label='FPT(stochastic simulations)', grid=False)
    ax[0].plot(tvec, [cfptd(t, T, A, p0)[0] for t in tvec], label='Analytical FPTD')
    ax[1].plot(tvec, [cfptd(t, T, A, p0)[1] for t in tvec], label='Analytical FPTD')
    plt.legend()
    plt.show()

