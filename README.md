# `fptools`: First-passage time tools for small- to medium-sized continuous-time, discrete-state systems

`fptools` is a set of simple tools for analyzing continuous-time, discrete-state
stochastic processes. It allows one to enter a set of transitions with their
associated transition rates and returns the relevant matrices suitable for
first-passage time analyses.

`fptools` contains three modules: fp, stochastic, and bayes. fp contains the
basic methods for calculating first-passage time analyses. stochastic allows
one to run stochastic simulations using the stochastic simulation algorithm.
bayes is a sketch of a method for doing Bayesian parameter fitting using these
models. It currently only supports fitting a single scaling parameter and
should be thought of as a proof-of-concept, not something that is ready for
full-blown parameter fitting of experimental data.

Note that `fptools` is meant for use with small- to medium-sized systems, up to
maybe several hundred states. `fptools` relies on NumPy routines for inverting
matrices and calculating matrix exponents, and while NumPy uses efficient
algorithms, they still scale rather poorly, so be careful when setting up
larger systems!

## Installation
**To do**

## How to use `fptools`
### Setting up a system
The fp module contains the function `make_sys`, which takes a multiline string
of transitions, e.g.,
```python
abc = """A, B, 3.0
B, A, 1.5
B, C, 2.7
"""
```
This string creates transitions from state A to state B, from B to A, and from
B to C with the transition rates 3.0, 1.5, and 2.7, respectively. Note that
this system contains one absorbing state, C: once it enters C, it cannot return
to the other, transient states. There must always be at least one absorbing
state, otherwise an exception is thrown.

Once you create the transitions, pass them to `make_sys`:
```python
idxW, idxT, idxA, W, T, A, transdims, absdims = make_sys(abc)
```
which will return dictionaries mapping state names to their numerical index in
the full, transient, and absorbing matrices; the three matrices themselves, and
two lists of indices that index where in the full transition matrix the
transient and absorbing states are.

### Carrying out first-passage time analyses
The fp module of `fptools` allows you to calculate mean exit and first-passage
times, splitting probabilites, and the full exit and first-passage time
probability distributions. The relevant functions take some combination of the
three matrices returned by `make_sys` and an initial probability distribution
over states. For example, to plot the exit time distribution of the system `abc`, run
```python
import matplotlib.pyplot as plt
tvec = np.linspace(0, 15, 200)  # Create a set of time points
plt.plot(tvec, [etc(t, T, A, p0) for t in tvec])
plt.show()
```

### Stochastic simulations
This is included as a sanity check mostly. `fptools` includes simple tools for
running stochastic simulations of the jump process using the stochastic
simulation algorithm (SSA) of Gillespie (1977). This is done using the `rep_ssa`
function in the `stochastic` module. Just pass the full transition matrix, the
index of the initial state, and the number of simulations to run, and it
returns a Pandas data frame with the exit/first-passage times and the absorbing
states for each run. To compare the exit time distribution to the distribution
of exit times from stochastic simulations, run:
```python
stoch = rep_ssa(W, 0, 1000)
stoch.hist(bins=50, density=True, label='Stochastic simulations')
plt.plot(tvec, [etc(t, T, A, p0) for t in tvec], label='Analytical density')
plt.legend()
plt.show()
```

### Bayesian parameter fitting
The `bayes` module is currently a proof-of-concept implementation. To use it,
you need a vector of data points to fit the model to. Currently, it only
supports fitting a single scaling parameter $`\tau`$ that scales the full
transition matrix. The idea here is to use the exit time distribution function
as the likelihood and truncated normal prior to calculate the posterior
probability distribution over $`\tau`$. The slowest part is calculating the
likelihood, but once that's done the `quick_posterior` function gives you the
posterior in a trivial amount of time.


