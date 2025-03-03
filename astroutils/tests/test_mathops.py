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

@pytest.mark.parametrize("nruns_shape, ndim, complex", [
    ((5,), 3, False),  # 5 real 3x3 matrices
    ((2, 4), 4, True),  # 2x4 batch of complex 4x4 matrices
    ((), 2, False),  # Single real 2x2 matrix
    ((3,), 5, True)  # 3 complex 5x5 matrices
])
def test_positive_definite_hermitian_matrix(nruns_shape, ndim, complex, positive_definite_hermitian_matrix):
    """Test if generated matrices are positive-definite and Hermitian."""
    
    # Generate matrices using the function under test
    matrices = MO.gen_random_positive_definite_hermitian_matrix(nruns_shape, ndim, complex)
    
    # Ensure the shape matches
    assert matrices.shape == nruns_shape + (ndim, ndim), "Output shape mismatch"

    # Check if matrices are Hermitian: A == A†
    assert NP.allclose(matrices, matrices.swapaxes(-2, -1).conj()), "Matrix is not Hermitian"

    # Check if matrices are positive definite: all eigenvalues must be > 0
    evals = NP.linalg.eigh(matrices)[0]  # Compute eigenvalues
    assert NP.all(evals > 0), "Matrix is not positive definite"

    # Compare with fixture-generated matrices (if needed)
    # Ensure eigenvalues of the fixture matrices are also positive
    fixture_matrices = positive_definite_hermitian_matrix
    fixture_evals = NP.linalg.eigh(fixture_matrices)[0]
    assert NP.all(fixture_evals > 0), "Fixture-generated matrix is not positive definite"

@pytest.mark.parametrize("nruns_shape, ndim, zero_indices, complex", [
    ((5,), 3, 0, False),  # 5 real 3x3 matrices with the largest singular value zeroed
    ((2, 4), 4, [1, 2], True),  # 2x4 batch of complex 4x4 matrices with two singular values zeroed
    ((), 2, [0], False),  # Single real 2x2 matrix with one singular value zeroed
    ((3,), 5, [0, 3], True)  # 3 complex 5x5 matrices with two singular values zeroed
])
def test_positive_semidefinite_hermitian_matrix(
    nruns_shape, ndim, zero_indices, complex, 
    positive_semidefinite_hermitian_matrix
):
    """Test if generated matrices are positive semi-definite and Hermitian with expected zeroed singular values."""

    # Generate matrices using the function under test
    matrices = MO.gen_random_positive_semidefinite_hermitian_matrix(nruns_shape, ndim, zero_indices, complex)

    # Ensure the shape matches
    assert matrices.shape == nruns_shape + (ndim, ndim), "Output shape mismatch"

    # Check if matrices are Hermitian: A == A†
    assert NP.allclose(matrices, matrices.swapaxes(-2, -1).conj()), "Matrix is not Hermitian"

    # Compute singular values
    _, S, _ = NP.linalg.svd(matrices)

    # Ensure the specified singular values are zero
    zero_indices_list = [zero_indices] if isinstance(zero_indices, int) else zero_indices
    assert NP.allclose(S[..., -len(zero_indices_list):],0), "Specified number of singular values are not zero"

    # Ensure at least one singular value is nonzero (to be positive semi-definite, not zero everywhere)
    assert NP.any(S > 0), "Matrix is entirely zero, expected positive semi-definite"

    # Ensure all singular values are non-negative (to be positive semi-definite, not negative anywhere)
    assert NP.all(S >= 0), "Matrix has negative eigenvalues, expected positive semi-definite"

    # Compare with fixture-generated matrices (if needed)
    fixture_matrices = positive_semidefinite_hermitian_matrix 
    # # Check that fixture matrices have expected zeroed singular values   
    _, fixture_S, _ = NP.linalg.svd(fixture_matrices)
    assert NP.allclose(fixture_S[..., -1],0), "Fixture-generated matrix does not match expected zeroed singular values"

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

########## Tests for multivariate Gaussian random variable generation #######

def test_multivariate_gaussian_2x2_covariance_no_mean():
    # Basic test with 2x2 covariance and no mean provided
    covariance = NP.array([[2+0j, 1+1j], [1-1j, 2+0j]], dtype=NP.complex128)
    result = MO.multivariate_gaussian(covariance)
    
    assert result.shape == (2,)
    assert NP.iscomplexobj(result), "Result should contain complex values."

def test_multivariate_gaussian_2x2_covariance_with_mean():
    # Test with 2x2 covariance and provided mean
    covariance = NP.array([[2+0j, 1+1j], [1-1j, 2+0j]], dtype=NP.complex128)
    mean = NP.array([1+0j, 2+0j], dtype=NP.complex128)
    result = MO.multivariate_gaussian(covariance, mean)
    
    assert result.shape == (2,)
    assert NP.iscomplexobj(result), "Result should contain complex values."

def test_multivariate_gaussian_batched_covariance():
    # Test with batched covariance and no nruns_shape provided
    covariance = NP.array([[[2+0j, 1+1j], [1-1j, 2+0j]], 
                           [[1+0j, 0+1j], [0-1j, 1+0j]]], dtype=NP.complex128)
    result = MO.multivariate_gaussian(covariance)
    
    assert result.shape == (2, 2), "Expected shape (2, 2) for the batched covariance."
    assert NP.iscomplexobj(result), "Result should contain complex values."

# Test broadcasting compatibility for mean vector and covariance
def test_multivariate_gaussian_batched():
    covariance = NP.array([[[2+0j, 0+0j], [0+0j, 3+0j]],
                           [[1+0j, 0+0j], [0+0j, 1+0j]]], dtype=NP.complex128)
    mean = NP.array([[1+0j, 1+0j], [0+0j, 0+0j]], dtype=NP.complex128)
    result = MO.multivariate_gaussian(covariance, mean=mean)
    assert result.shape == (2, 2), "Batched test failed: incorrect shape"

# Test covariance matrix structure preservation after sampling
def test_multivariate_gaussian_covariance_preservation():
    covariance = NP.array([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=NP.complex128)
    samples = NP.array([MO.multivariate_gaussian(covariance) for _ in range(5000)])
    sample_covariance = NP.cov(samples.T)
    assert NP.allclose(sample_covariance, covariance, atol=0.05), "Covariance preservation test failed"

# Test if the function maintains covariance structure after generating samples
def test_covariance_structure():
    covariance = NP.array([[2+0j, 1+1j], [1-1j, 2+0j]], dtype=NP.complex128)
    samples = NP.array([MO.multivariate_gaussian(covariance) for _ in range(10000)])
    sample_covariance = NP.cov(samples.T)
    assert NP.allclose(sample_covariance, covariance, atol=0.1), "Generated covariance structure does not match"

# Test if broadcasting works properly when mean and size are given in different dimensions
def test_multivariate_gaussian_broadcast_mean_size():
    covariance = NP.array([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=NP.complex128)
    mean = NP.array([3+0j, 3+0j], dtype=NP.complex128)
    size = (4,)
    result = MO.multivariate_gaussian(covariance, mean=mean, nruns_shape=size)
    assert result.shape == (4, 2), "Broadcast mean and size test failed: incorrect shape"

def test_output_shape():
    """Test if output shape matches expected shape."""
    covariance = NP.array([[2+0j, 1+1j], [1-1j, 2+0j]], dtype=NP.complex128)
    assert MO.multivariate_gaussian(covariance).shape == (2,)
    assert MO.multivariate_gaussian(covariance, nruns_shape=(5,)).shape == (5, 2)
    
    batch_cov = NP.array([
        [[2+0j, 1+1j], [1-1j, 2+0j]],
        [[1+0j, 0+1j], [0-1j, 1+0j]]
    ], dtype=NP.complex128)
    assert MO.multivariate_gaussian(batch_cov, nruns_shape=(3, 4)).shape == (3, 4, 2, 2)

def test_mean_application():
    """Test if the mean is correctly applied."""
    covariance = NP.eye(3, dtype=NP.complex128)
    mean = NP.array([1+1j, 2+2j, 3+3j], dtype=NP.complex128)
    samples = MO.multivariate_gaussian(covariance, mean=mean, nruns_shape=(100,))
    assert samples.shape == (100, 3)
    assert NP.allclose(samples.mean(axis=0), mean, atol=0.5)  # Allow some tolerance due to randomness


def test_zero_mean_unit_variance():
    """Test if the function generates zero-mean unit-variance data when using identity covariance."""
    NP.random.seed(42)  # For reproducibility
    covariance = NP.eye(3, dtype=NP.complex128)
    samples = MO.multivariate_gaussian(covariance, nruns_shape=(10000,))
    assert NP.allclose(samples.mean(axis=0), 0, atol=0.1)
    assert NP.allclose(NP.cov(samples.T), NP.eye(3), atol=0.1)

# def test_invalid_nruns_shape():
#     """Test if invalid nruns_shape raises an error."""
#     covariance = NP.eye(2, dtype=NP.complex128)
#     with pytest.raises(ValueError):
#         MO.multivariate_gaussian(covariance, nruns_shape=(2, 3, 4))  # Incompatible with covariance shape

def test_non_psd_handling():
    """Test if the function properly handles non-positive-semidefinite covariance."""
    covariance = NP.array([[1+0j, 2+0j], [2+0j, 1+0j]], dtype=NP.complex128)  # Not PSD
    samples = MO.multivariate_gaussian(covariance, nruns_shape=(100,))
    assert samples.shape == (100, 2)
    # Check if covariance of generated samples is close to expected
    assert not NP.allclose(NP.cov(samples.T), covariance, atol=0.1)

def test_broadcasting():
    """Test if broadcasting works correctly when nruns_shape has extra dimensions."""
    covariance = NP.eye(3, dtype=NP.complex128)
    samples = MO.multivariate_gaussian(covariance, nruns_shape=(2, 3))
    assert samples.shape == (2, 3, 3)

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

