from __future__ import print_function, division, unicode_literals, absolute_import

import warnings 
import pytest 

import numpy as NP
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .. import mathops as MO

def test_reverse():
    n1, n2, n3 = 2, 3, 4
    inp = NP.arange(n1*n2*n3)
    NP.testing.assert_allclose(MO.reverse(inp), inp[::-1])

    inp = inp.reshape(n1,n2,n3)
    out_expected = NP.asarray([[[8, 9, 10, 11],
                                [4, 5, 6, 7],
                                [0, 1, 2, 3]],
                               [[20, 21, 22, 23],
                                [16, 17, 18, 19],
                                [12, 13, 14, 15]]])
    NP.testing.assert_allclose(MO.reverse(inp, axis=1), out_expected)

def test_binned_statistic():
    inp = NP.asarray([0.3, 0.5, 1.25, 2.25, 2.5, 2.75])
    val = NP.arange(inp.size)
    bins = NP.arange(4)
    expected_count = NP.asarray([2, 1, 3])
    expected_means = NP.asarray([0.5, 2.0, 4.0])

    count, be, bn, ri = MO.binned_statistic(inp, statistic='count', bins=bins)
    NP.testing.assert_allclose(count, expected_count)

    mean_vals, be, bn, ri = MO.binned_statistic(inp, values=val, statistic='mean', bins=bins)
    NP.testing.assert_allclose(mean_vals, expected_means)

def test_rms():
    n1, n2 = 3, 2
    inp = NP.arange(n1*n2).reshape(n1,n2)
    expected_rms = NP.std(inp, axis=1, keepdims=True)

    NP.testing.assert_allclose(MO.rms(inp, axis=1), expected_rms)
    
def test_hermitian_input_type():
    with pytest.raises(TypeError, match="Input array inparr must be a numpy array"):
        MO.hermitian([1, 2, 3])

    with pytest.raises(TypeError, match="Input axes must be a list, tuple, or numpy array"):
        MO.hermitian(NP.array([[1, 2], [3, 4]]), axes="invalid_axes")

def test_hermitian_input_shape():
    input_array = NP.array([1, 2, 3])
    result = MO.hermitian(input_array)
    assert result.shape == (3,1), "Hermitian shape mismatch for 1D input"

def test_hermitian_axes_type():
    with pytest.raises(ValueError, match="Input axes must be a two-element list, tuple, or numpy array"):
        MO.hermitian(NP.array([[1, 2], [3, 4]]), axes=[0])

def test_hermitian_axes_value():
    with pytest.raises(ValueError, match="The two entries in axes cannot be the same"):
        MO.hermitian(NP.array([[1, 2], [3, 4]]), axes=(0, 0))

def test_hermitian():
    input_array = NP.array([[1+2j, 2 + 1j], [3 - 2j, 4-3j]])
    result = MO.hermitian(input_array, axes=(0, 1))
    expected_result = input_array.T.conj()
    assert NP.allclose(result, expected_result), "Numerical Hermitian check failed"

def test_hat_input_type():
    with pytest.raises(TypeError, match="Input array inparr must be a numpy array"):
        MO.hat([1,2,3])

    with pytest.raises(TypeError, match="Input axes must be a list, tuple, or numpy array"):
        MO.hat(NP.array([[1,2], [3,4]]), axes="invalid_axes")

def test_hat_input_shape():
    input_array = NP.array([[1,2,3], [4,5,3]])
    with pytest.raises(ValueError, match="The axes of inversion must be square in shape"):
        MO.hat(input_array, axes=None)

def test_hat_axes_type():
    with pytest.raises(ValueError, match="Input axes must be a two-element list, tuple, or numpy array"):
        MO.hat(NP.array([[1,2], [3,4]]), axes=[0])

def test_hat_axes_value():
    with pytest.raises(ValueError, match="The two entries in axes cannot be the same"):
        MO.hat(NP.array([[1,2], [3,4]]), axes=(0, 0))

def test_hat_numerical():
    input_array = NP.array([[1+2j, 2 + 1j], [3 - 2j, 4-3j]])
    result = MO.hat(input_array, axes=(0, 1))
    hermitian_result = MO.hermitian(input_array, axes=(0, 1))
    expected_result = NP.linalg.inv(hermitian_result)
    assert NP.allclose(result, expected_result), "Numerical Hat operation check failed"

def test_sqrt_positive_definite_hermitian_matrix(positive_definite_hermitian_matrix):
    sqrt_matrix = MO.sqrt_matrix_factorization(positive_definite_hermitian_matrix)
    assert NP.allclose(sqrt_matrix @ NP.swapaxes(sqrt_matrix.conj(),-2,-1), positive_definite_hermitian_matrix), "Square root factorization failed for positive-definite Hermitian matrix."

def test_sqrt_positive_semi_definite_hermitian_matrix(positive_semidefinite_hermitian_matrix):
    """Test square root factorization for a positive semi-definite Hermitian matrix."""
    with warnings.catch_warnings(record=True) as w:
        sqrt_matrix = MO.sqrt_matrix_factorization(positive_semidefinite_hermitian_matrix)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "not positive semi-definite" in str(w[-1].message)
    assert NP.allclose(sqrt_matrix @ NP.swapaxes(sqrt_matrix.conj(),-2,-1), positive_semidefinite_hermitian_matrix), "Square root factorization failed for positive-semi-definite Hermitian matrix."

def test_sqrt_non_hermitian_matrix(non_hermitian_matrix):
    """Test that a non-Hermitian matrix raises a ValueError."""
    with pytest.raises(ValueError, match="Input matrix is not Hermitian"):
        MO.sqrt_matrix_factorization(non_hermitian_matrix)

###### Tests for unscented transform ########

# # Define a simple linear function for testing
# def linear_func(x: NDArray[NP.float64]) -> NDArray[NP.float64]:
#     return 2.0 * x + 1.0  # A linear transformation y = 2x + 1

# # Define an identity function (should return the input)
# def identity_func(x: NDArray[NP.float64]) -> NDArray[NP.float64]:
#     return x

# # Define a non-linear function for more complex testing
# def non_linear_func(x: NDArray[NP.float64]) -> NDArray[NP.float64]:
#     return x**2

# # Test for linear transformation
# def test_unscented_transform_linear_func(positive_definite_hermitian_matrix):
#     # mean = NP.array([2.0, 3.0])
#     # covariance = NP.array([[1.0, 0.5], [0.5, 1.5]])
#     ndim_inp = positive_definite_hermitian_matrix.ndim
#     preinds = (0,)*(ndim_inp-2) # (0,0,...)
#     covariance = positive_definite_hermitian_matrix.real[preinds]
#     mean = NP.random.normal(size=(covariance.shape[-1],))

#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, linear_func)

#     # Expected transformed mean
#     expected_mean = 2 * mean + 1  # Since the transformation is linear
#     assert_almost_equal(transformed_mean, expected_mean, decimal=4)

#     # Expected transformed covariance
#     expected_covariance = 2 * 2 * covariance  # Linear transformation scales covariance by the square of the factor (2)
#     assert_almost_equal(transformed_covariance, expected_covariance, decimal=6)

# # Test for identity transformation (should return the original mean and covariance)
# def test_unscented_transform_identity_func():
#     mean = NP.array([0.0, 1.0])
#     covariance = NP.array([[1.0, 0.0], [0.0, 1.0]])

#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, identity_func)

#     # Check that the transformed mean and covariance are the same as the input
#     assert_almost_equal(transformed_mean, mean, decimal=6)
#     assert_almost_equal(transformed_covariance, covariance, decimal=6)

# # Test for higher-dimensional input (vectorization)
# def test_unscented_transform_vectorized_input(positive_definite_hermitian_matrix):
#     # mean = NP.array([[2.0, 3.0], [4.0, 5.0]])
#     # covariance = NP.array([[[1.0, 0.5], [0.5, 1.5]], [[2.0, 0.3], [0.3, 2.5]]])
#     covariance = positive_definite_hermitian_matrix.real
#     nruns_shape = covariance.shape[:-2]
#     mean = NP.random.normal(size=nruns_shape+(covariance.shape[-1],))

#     # Linear transformation applied to each element in vectorized fashion
#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, linear_func)

#     # Expected transformed mean and covariance for each set
#     expected_mean = 2 * mean + 1
#     expected_covariance = 2 * 2 * covariance

#     assert_array_almost_equal(transformed_mean, expected_mean, decimal=6)
#     assert_array_almost_equal(transformed_covariance, expected_covariance, decimal=6)

# # Test for non-linear transformation
# def test_unscented_transform_non_linear_func():
#     mean = NP.array([0.5, 1.0])
#     covariance = NP.array([[0.1, 0.05], [0.05, 0.2]])

#     # Apply a non-linear transformation
#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, non_linear_func)

#     # Non-linear transformation results should be tested based on expected properties of the transformation
#     # For now, just check the shapes and sanity of the outputs
#     assert transformed_mean.shape == mean.shape
#     assert transformed_covariance.shape == covariance.shape

# # Test for 3D covariance (broadcasting test)
# def test_unscented_transform_3d_covariance():
#     mean = NP.array([1.0, 2.0])
#     covariance = NP.array([[[1.0, 0.2], [0.2, 1.5]], [[1.0, 0.3], [0.3, 1.2]]])

#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, linear_func)

#     expected_mean = 2 * mean + 1
#     expected_covariance = 2 * 2 * covariance

#     assert_almost_equal(transformed_mean, expected_mean, decimal=6)
#     assert_almost_equal(transformed_covariance, expected_covariance, decimal=6)

# # Test for exception handling in sqrtm fallback
# def test_unscented_transform_sqrtm_fallback():
#     mean = NP.array([0.0, 1.0])
#     covariance = NP.array([[1.0, 2.0], [2.0, 1.0]])  # This matrix will cause sqrtm to fail

#     # The function should fallback to the SVD-based square root when sqrtm fails
#     transformed_mean, transformed_covariance = MO.unscented_transform(mean, covariance, identity_func)

#     # Check that the transformed values have the correct shape and are finite
#     assert transformed_mean.shape == mean.shape
#     assert transformed_covariance.shape == covariance.shape
#     assert NP.all(NP.isfinite(transformed_covariance))

