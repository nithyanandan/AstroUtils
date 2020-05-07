from __future__ import print_function, division, unicode_literals, absolute_import
from astroutils import mathops as OPS
import numpy as NP

def test_reverse():
    n1, n2, n3 = 2, 3, 4
    inp = NP.arange(n1*n2*n3)
    NP.testing.assert_allclose(OPS.reverse(inp), inp[::-1])

    inp = inp.reshape(n1,n2,n3)
    out_expected = NP.asarray([[[8, 9, 10, 11],
                                [4, 5, 6, 7],
                                [0, 1, 2, 3]],
                               [[20, 21, 22, 23],
                                [16, 17, 18, 19],
                                [12, 13, 14, 15]]])
    NP.testing.assert_allclose(OPS.reverse(inp, axis=1), out_expected)

def test_binned_statistic():
    inp = NP.asarray([0.3, 0.5, 1.25, 2.25, 2.5, 2.75])
    val = NP.arange(inp.size)
    bins = NP.arange(4)
    expected_count = NP.asarray([2, 1, 3])
    expected_means = NP.asarray([0.5, 2.0, 4.0])

    count, be, bn, ri = OPS.binned_statistic(inp, statistic='count', bins=bins)
    NP.testing.assert_allclose(count, expected_count)

    mean_vals, be, bn, ri = OPS.binned_statistic(inp, values=val, statistic='mean', bins=bins)
    NP.testing.assert_allclose(mean_vals, expected_means)

def test_rms():
    n1, n2 = 3, 2
    inp = NP.arange(n1*n2).reshape(n1,n2)
    expected_rms = NP.std(inp, axis=1, keepdims=True)

    NP.testing.assert_allclose(OPS.rms(inp, axis=1), expected_rms)
    
