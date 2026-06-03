import pytest
import numpy as np

from pychemelt.utils.math import (
    first_derivative_savgol,
    solve_one_root_quadratic,
    solve_one_root_depressed_cubic,
    quadratic_baseline_only_temp,
    aic_bic_eff,
    extended_bic
)


def test_first_derivative_savgol_error():

    x = [1, 3, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # Raise value error if x is not evenly spaced
    with pytest.raises(ValueError):
        first_derivative_savgol(x,y)

def test_first_derivative_savgol_polyorder_error():

    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # Raise value error if the window is too short
    with pytest.raises(ValueError):
        first_derivative_savgol(x,y,window_length=1)


def test_first_derivative_savgol_accuracy():
    # Test linear function y = 2x
    x = np.linspace(0, 10, 101)  # dx = 0.1
    y = 2 * x

    # window_length=2 means n = 2/0.1 // 2 * 2 + 1 = 21
    deriv = first_derivative_savgol(x, y, window_length=2, polyorder=2)

    # The middle part should be exactly 2.0
    # Edges might be affected by mode="nearest"
    assert np.mean(deriv[10:-10]) == pytest.approx(2.0)

    # Test with different spacing
    x2 = np.linspace(0, 10, 201)  # dx = 0.05
    y2 = 2 * x2
    deriv2 = first_derivative_savgol(x2, y2, window_length=2, polyorder=2)
    assert np.mean(deriv2[20:-20]) == pytest.approx(2.0)

def test_solve_one_root_quadratic():

    assert solve_one_root_quadratic(3, 2, -1) == pytest.approx(1/3)

    #division by 0
    assert np.isnan(solve_one_root_quadratic(2, -2, 0))

    assert solve_one_root_quadratic(2, 5, 1.125) == pytest.approx(-0.25)


def test_solve_one_root_depressed_cubic():

    assert solve_one_root_depressed_cubic(2, 2) == pytest.approx(-0.77092, abs=1e-4)


def test_quadratic_baseline_only_temp():

    dt = np.array([0, 1, 2, 3])
    a, b, c = 1, -0.5, 0.1

    expected = a + b*dt + c*dt**2
    result = quadratic_baseline_only_temp(dt, a, b, c)

    assert np.allclose(result, expected)


def test_extended_bic_equals_bic_when_gamma_zero():
    """
    Test that extended BIC equals standard BIC when gamma=0.
    """
    # Create a mock result object with necessary attributes
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    # Test with various parameter combinations
    test_cases = [
        (100.0, 50, 3),   # chisqr=100, neff=50, k=3
        (250.5, 100, 5),  # chisqr=250.5, neff=100, k=5
        (50.0, 200, 10),  # chisqr=50, neff=200, k=10
    ]
    
    for chisqr, neff, k in test_cases:
        result = MockResult(chisqr, k)
        
        # Get BIC from aic_bic_eff (returns tuple: AIC, BIC)
        aic, bic = aic_bic_eff(result, neff)
        
        # Get EBIC with gamma=0 from extended_bic
        ebic_gamma_zero = extended_bic(result, neff, gamma=0.0)
        
        # They should be equal
        assert ebic_gamma_zero == pytest.approx(bic), \
            f"EBIC(gamma=0) = {ebic_gamma_zero} should equal BIC = {bic}"


def test_extended_bic_increases_with_gamma():
    """EBIC should increase as gamma increases (stronger penalty)."""
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    result = MockResult(100.0, 5)
    neff = 50
    
    ebic_0 = extended_bic(result, neff, gamma=0.0)
    ebic_25 = extended_bic(result, neff, gamma=0.25)
    ebic_50 = extended_bic(result, neff, gamma=0.5)
    ebic_75 = extended_bic(result, neff, gamma=0.75)
    ebic_100 = extended_bic(result, neff, gamma=1.0)
    
    # EBIC should be monotonically increasing with gamma
    assert ebic_0 < ebic_25 < ebic_50 < ebic_75 < ebic_100


def test_extended_bic_penalizes_complexity():
    """EBIC should increase with more parameters for same chi-squared."""
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    neff = 100
    chisqr = 50.0
    gamma = 0.5
    
    # Same fit quality, different number of parameters
    result_3params = MockResult(chisqr, 3)
    result_5params = MockResult(chisqr, 5)
    result_10params = MockResult(chisqr, 10)
    
    ebic_3 = extended_bic(result_3params, neff, gamma)
    ebic_5 = extended_bic(result_5params, neff, gamma)
    ebic_10 = extended_bic(result_10params, neff, gamma)
    
    # More parameters → higher EBIC (worse)
    assert ebic_3 < ebic_5 < ebic_10


def test_extended_bic_rewards_better_fit():
    """EBIC should decrease with better fit (lower chi-squared)."""
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    neff = 100
    k = 5
    gamma = 0.5
    
    result_bad = MockResult(200.0, k)
    result_medium = MockResult(100.0, k)
    result_good = MockResult(50.0, k)
    
    ebic_bad = extended_bic(result_bad, neff, gamma)
    ebic_medium = extended_bic(result_medium, neff, gamma)
    ebic_good = extended_bic(result_good, neff, gamma)
    
    # Better fit (lower chi-squared) → lower EBIC (better)
    assert ebic_bad > ebic_medium > ebic_good


def test_extended_bic_edge_cases():
    """Test EBIC with edge cases: k=0, k=1, k close to neff."""
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    # k=0 should not cause errors (extended_penalty=0)
    result_k0 = MockResult(100.0, 0)
    ebic_k0 = extended_bic(result_k0, neff=50, gamma=0.5)
    assert not np.isnan(ebic_k0) and not np.isinf(ebic_k0)
    
    # k=1 should work
    result_k1 = MockResult(100.0, 1)
    ebic_k1 = extended_bic(result_k1, neff=50, gamma=0.5)
    assert not np.isnan(ebic_k1) and not np.isinf(ebic_k1)
    
    # k close to neff should not crash (though not realistic)
    result_k_large = MockResult(100.0, 49)
    ebic_k_large = extended_bic(result_k_large, neff=50, gamma=0.5)
    assert not np.isnan(ebic_k_large) and not np.isinf(ebic_k_large)


def test_extended_bic_formula():
    """Test EBIC formula matches expected calculation."""
    from scipy.special import gammaln
    
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    chisqr = 100.0
    neff = 50
    k = 3
    gamma = 0.5
    
    result = MockResult(chisqr, k)
    ebic = extended_bic(result, neff, gamma)
    
    # Calculate expected EBIC manually
    bic_term = neff * np.log(chisqr / neff) + np.log(neff) * k
    log_binom = gammaln(neff + 1) - gammaln(k + 1) - gammaln(neff - k + 1)
    expected_ebic = bic_term + 2 * gamma * log_binom
    
    assert ebic == pytest.approx(expected_ebic)


def test_extended_bic_more_conservative_than_bic():
    """EBIC should be >= BIC for gamma > 0."""
    class MockResult:
        def __init__(self, chisqr, nvarys):
            self.chisqr = chisqr
            self.nvarys = nvarys
    
    test_cases = [
        (100.0, 50, 3),
        (250.5, 100, 5),
        (50.0, 200, 10),
    ]
    
    for chisqr, neff, k in test_cases:
        result = MockResult(chisqr, k)
        
        _, bic = aic_bic_eff(result, neff)
        ebic_05 = extended_bic(result, neff, gamma=0.5)
        ebic_10 = extended_bic(result, neff, gamma=1.0)
        
        # EBIC should be >= BIC when gamma > 0
        assert ebic_05 >= bic
        assert ebic_10 >= bic
        assert ebic_10 > ebic_05  # Higher gamma → higher penalty
