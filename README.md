# FPTools: First-passage time tools for small- to medium-sized continuous-time, discrete-state systems

FPTools is a set of simple tools for analyzing continuous-time, discrete-state
stochastic processes. It allows one to enter a set of transitions with their
associated transition rates and returns the relevant matrices suitable for
first-passage time analyses.

FPTools contains three modules: fp, stochastic, and bayes. fp contains the
basic methods for calculating first-passage time analyses. stochastic allows
one to run stochastic simulations using the stochastic simulation algorithm.
bayes is a sketch of a method for doing Bayesian parameter fitting using these
models. It currently only supports fitting a single scaling parameter and
should be thought of as a proof-of-concept, not something that is ready for
full-blown parameter fitting of experimental data.

## How to use FPTools
### Setting up a system
The fp module contains the function `make_sys`, which takes a multiline string
of transitions, e.g.,

   """A, B, 3.0
   B, A, 1.5
   B, C, 2.7
   """

This string creates transitions from state A to state B, from B to A, and from
B to C with the transition rates 3.0, 1.5, and 2.7, respectively. Note that
this system contains one absorbing state, C: once it enters C, it cannot return
to the other, transient states. There must always be at least one absorbing
state, otherwise an exception is thrown.

Once you create the transitions, pass them to `make_sys`, which will return
dictionaries mapping state names to their numerical index in the full,
transient, and absorbing matrices; the three matrices themselves, and two lists
of indices that index where in the full transition matrix the transient and
absorbing states are.

### Carrying out first-passage time analyses
The fp module of FPTools allows you to calculate mean exit and first-passage
times, splitting probabilites, and the full exit and first-passage time
probability distributions. The relevant functions take some combination of the
three matrices returned by `make_sys` and an initial probability distribution
over states.


