from __future__ import print_function, division, unicode_literals, absolute_import

import warnings 
import pytest 

import numpy as NP
from .. import mathops as OPS

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
    
def test_hermitian_input_type():
    with pytest.raises(TypeError, match="Input array inparr must be a numpy array"):
        OPS.hermitian([1, 2, 3])

    with pytest.raises(TypeError, match="Input axes must be a list, tuple, or numpy array"):
        OPS.hermitian(NP.array([[1, 2], [3, 4]]), axes="invalid_axes")

def test_hermitian_input_shape():
    input_array = NP.array([1, 2, 3])
    result = OPS.hermitian(input_array)
    assert result.shape == (3,1), "Hermitian shape mismatch for 1D input"

def test_hermitian_axes_type():
    with pytest.raises(ValueError, match="Input axes must be a two-element list, tuple, or numpy array"):
        OPS.hermitian(NP.array([[1, 2], [3, 4]]), axes=[0])

def test_hermitian_axes_value():
    with pytest.raises(ValueError, match="The two entries in axes cannot be the same"):
        OPS.hermitian(NP.array([[1, 2], [3, 4]]), axes=(0, 0))

def test_hermitian():
    input_array = NP.array([[1+2j, 2 + 1j], [3 - 2j, 4-3j]])
    result = OPS.hermitian(input_array, axes=(0, 1))
    expected_result = input_array.T.conj()
    assert NP.allclose(result, expected_result), "Numerical Hermitian check failed"

def test_hat_input_type():
    with pytest.raises(TypeError, match="Input array inparr must be a numpy array"):
        OPS.hat([1,2,3])

    with pytest.raises(TypeError, match="Input axes must be a list, tuple, or numpy array"):
        OPS.hat(NP.array([[1,2], [3,4]]), axes="invalid_axes")

def test_hat_input_shape():
    input_array = NP.array([[1,2,3], [4,5,3]])
    with pytest.raises(ValueError, match="The axes of inversion must be square in shape"):
        OPS.hat(input_array, axes=None)

def test_hat_axes_type():
    with pytest.raises(ValueError, match="Input axes must be a two-element list, tuple, or numpy array"):
        OPS.hat(NP.array([[1,2], [3,4]]), axes=[0])

def test_hat_axes_value():
    with pytest.raises(ValueError, match="The two entries in axes cannot be the same"):
        OPS.hat(NP.array([[1,2], [3,4]]), axes=(0, 0))

def test_hat_numerical():
    input_array = NP.array([[1+2j, 2 + 1j], [3 - 2j, 4-3j]])
    result = OPS.hat(input_array, axes=(0, 1))
    hermitian_result = OPS.hermitian(input_array, axes=(0, 1))
    expected_result = NP.linalg.inv(hermitian_result)
    assert NP.allclose(result, expected_result), "Numerical Hat operation check failed"

def test_sqrt_positive_definite_hermitian_matrix(positive_definite_hermitian_matrix):
    sqrt_matrix = OPS.sqrt_matrix_factorization(positive_definite_hermitian_matrix)
    assert NP.allclose(sqrt_matrix @ NP.swapaxes(sqrt_matrix.conj(),-2,-1), positive_definite_hermitian_matrix), "Square root factorization failed for positive-definite Hermitian matrix."

def test_sqrt_positive_semi_definite_hermitian_matrix(positive_semidefinite_hermitian_matrix):
    """Test square root factorization for a positive semi-definite Hermitian matrix."""
    with warnings.catch_warnings(record=True) as w:
        sqrt_matrix = OPS.sqrt_matrix_factorization(positive_semidefinite_hermitian_matrix)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "not positive semi-definite" in str(w[-1].message)
    assert NP.allclose(sqrt_matrix @ NP.swapaxes(sqrt_matrix.conj(),-2,-1), positive_semidefinite_hermitian_matrix), "Square root factorization failed for positive-semi-definite Hermitian matrix."

def test_sqrt_non_hermitian_matrix(non_hermitian_matrix):
    """Test that a non-Hermitian matrix raises a ValueError."""
    with pytest.raises(ValueError, match="Input matrix is not Hermitian"):
        OPS.sqrt_matrix_factorization(non_hermitian_matrix)

