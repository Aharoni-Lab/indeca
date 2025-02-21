import pytest
import numpy as np
from minian_bin.utils import norm, scal_lstsq, scal_like, enumerated_product

class TestNormalization:
    def test_norm_basic(self):
        """Test basic normalization."""
        a = np.array([1, 2, 3, 4, 5])
        result = norm(a)
        assert np.allclose(result, np.array([0, 0.25, 0.5, 0.75, 1.0]))
    
    def test_norm_constant(self):
        """Test normalization with constant array."""
        a = np.array([1, 1, 1])
        result = norm(a)
        assert np.allclose(result, np.array([0, 0, 0]))
    
    def test_norm_with_nan(self):
        """Test normalization with NaN values."""
        a = np.array([1, np.nan, 3, 4])
        result = norm(a)
        assert np.isnan(result[1])
        assert not np.any(np.isnan(result[[0, 2, 3]]))

class TestScaling:
    def test_scal_lstsq_1d(self):
        """Test least squares scaling with 1D arrays."""
        a = np.array([1, 2, 3])
        b = np.array([2, 4, 6])
        result = scal_lstsq(a, b)
        assert np.isclose(result, 2.0)
    
    def test_scal_lstsq_2d(self):
        """Test least squares scaling with 2D array."""
        a = np.array([[1], [2], [3]])
        b = np.array([2, 4, 6])
        result = scal_lstsq(a, b)
        assert np.isclose(result, 2.0)

    def test_scal_like_zero_center(self):
        """Test scaling with zero centering."""
        src = np.array([1, 2, 3])
        tgt = np.array([0, 5, 10])
        result = scal_like(src, tgt, zero_center=True)
        assert np.allclose(result, np.array([5, 10, 15]))
    
    def test_scal_like_no_zero_center(self):
        """Test scaling without zero centering."""
        src = np.array([1, 2, 3])
        tgt = np.array([0, 5, 10])
        result = scal_like(src, tgt, zero_center=False)
        assert np.allclose(result, np.array([0, 5, 10]))

class TestEnumeratedProduct:
    def test_basic_product(self):
        """Test basic enumerated product functionality."""
        result = list(enumerated_product([1, 2], ['a', 'b']))
        expected = [
            ((0, 0), (1, 'a')),
            ((0, 1), (1, 'b')),
            ((1, 0), (2, 'a')),
            ((1, 1), (2, 'b'))
        ]
        assert result == expected
    
    def test_empty_input(self):
        """Test enumerated product with empty input."""
        result = list(enumerated_product([], []))
        assert result == [] 