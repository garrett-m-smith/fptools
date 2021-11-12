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


#from fp import *
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm


def likelihood(tau, data, T, A, p0):
    """Takes a free scaling parameter tau, a vector of data points (observed
    exit/reading times), and the transient and absorbing matrices T and A along
    with an initial condition vector p0. Returns a function from tau values to
    likelihood values.
    """
    print("NOTE: This implementation does not guarantee correct transition matrices!")
    vec = [etd(t, tau*T, tau*A, p0) for t in data]
    #return np.prod([etd(t, tau*T, tau*A, p0) for t in data])  # Works
    return np.exp(sum(np.log(vec)))


def prior(tau, prmean=5.0, prsd=1.0):
    """The prior distribution over the scaling parameter tau. Assumes a
    truncated normal distribution (>0.001). Requires a mean and standard
    deviation.
    """
    return np.clip(a=norm(prmean, prsd).pdf(tau), a_min=0.001, a_max=None)


def calc_evidence(tau, prmean, prsd, data, T, A, p0, domain):
    """Calculates the evidence, average likelihood or average probability of
    the data given prior distribution and a likelihood.
    """
    area, _ = quad(lambda ta: likelihood(ta, data, T, A, p0)*prior(ta, prmean, prsd),
                   domain[0], domain[1], epsabs=1e-4)
    print('Evidence: {}'.format(area))
    return area


def quick_posterior(prvec, lkvec, normalize=True):
    """Given a vector of prior probabilities and likelihoods, returns a
    (normalized) vector of posterior probabilities.
    """
    unnorm = prvec * lkvec
    if normalize:
        return unnorm / np.trapz(unnorm)
    else:
        return unnorm


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #data = [0.286, 0.297, 0.275]
    data = np.clip(np.random.normal(0.4, 0.1, 50), a_min=0.001, a_max=None)  # Approx. typical
    #truetau = 3.5
    T = np.array([-2., 1, 1, 1, -2, 0, 1, 0, -1]).reshape(3,3)
    A = np.array([0, 1, 0])
    p0 = np.array([1, 0, 0])

    #plt.plot([etd(t, T, A, p0) for t in np.linspace(0, 10, 100)])
    #plt.show()

    #truefpt = meanvar(truetau*T, p0)[0]
    #data = [truefpt]  # Works
    #data = np.clip(np.random.normal(truefpt, 0.5, 50), a_min=0.001, a_max=None)
    plt.hist(data); plt.title('Data'); plt.show()
    
    tauvec = np.linspace(0.01, 20, 200)
    prmean = max(tauvec) / 2
    prsd = max(tauvec) / 4
    pr = np.array([prior(tau, prmean, prsd) for tau in tauvec])
    plt.plot(tauvec, pr)
    plt.title('Prior')
    plt.xlabel('tau')
    plt.show()

    lk = np.array([likelihood(tau, data, T, A, p0) for tau in tqdm(tauvec)])
    plt.plot(tauvec, lk)
    plt.title('Likelihood')
    plt.xlabel('tau')
    plt.show()

    #pst = [posterior(tau, prmean, prsd, data, T, A, p0, normalize=[0.01, 50]) for tau in tauvec]
    pst = quick_posterior(pr, lk)
    assert np.isclose(np.trapz(pst), 1.0), "Posterior doesn't integrate to 1.0"
    plt.plot(tauvec, pst)
    plt.title('Posterior')
    plt.xlabel('tau')
    plt.show()

    taumax = tauvec[np.argmax(pst)]
    print('True mean: {}\nRecovered MAP {}'.format(np.mean(data),
        meanvar(taumax*T, p0)[0]))


