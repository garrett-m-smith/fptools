from unittest import TestCase
from fptools import fp, stochastic
import numpy as np
from scipy.stats import erlang


trans = """a, b, 1.0
b, c, 1.0
"""
_, _, _, W, T, A, _, _ = fp.make_sys(trans, scale_rates=False)
p0 = [1.0, 0]
tvec = np.linspace(0, 10, 100)

class TestErlang(TestCase):
    def test_pdf(self):
        fpvals = [fp.etd(t, T, A, p0) for t in tvec]
        erlvals = [erlang.pdf(x=t, a=2) for t in tvec]
        self.assertTrue(np.allclose(fpvals, erlvals, atol=0.01))

    def test_moments(self):
        erlmean, erlvar = erlang.stats(a=2, moments='mv')
        fpmean, fpvar = fp.meanvar(T, p0)
        self.assertTrue(np.allclose([erlmean, erlvar], [fpmean, fpvar]))
    
    def test_stochastic(self):
        erlmean, erlvar = erlang.stats(a=2, moments='mv')
        stoch = stochastic.rep_ssa(W, initidx=0, n=5000)
        print(erlmean, erlvar, [stoch['fpt'].mean(), stoch['fpt'].var()])
        self.assertTrue(np.allclose([erlmean, erlvar], [stoch['fpt'].mean(), stoch['fpt'].var()], atol=0.1))


if __name__ == "__main__":
    unittest.main()