from ..Quadrature import Quadrature1D
import pytest

def test_weights_linear():
    q1 = Quadrature1D(order=1)
    assert helper_check_integrand(expected=-9.0, quad=q1, funct=linear_function)
    
    q2 = Quadrature1D(order=2)
    assert helper_check_integrand(expected=-9.0, quad=q2, funct=linear_function)
    
    q3 = Quadrature1D(order=3)
    assert helper_check_integrand(expected=-9.0, quad=q3, funct=linear_function)

def test_weights_quadratic():
    q1 = Quadrature1D(order=1)
    assert helper_check_integrand(expected=-13.2, quad=q1, funct=quadratic_function)
    
    q2 = Quadrature1D(order=2)
    assert helper_check_integrand(expected=-13.2, quad=q2, funct=quadratic_function)
    
    q3 = Quadrature1D(order=3)
    assert helper_check_integrand(expected=-13.2, quad=q3, funct=quadratic_function)

def test_weights_cubic():
    q1 = Quadrature1D(order=1)
    assert helper_check_integrand(expected=-13.2, quad=q1, funct=cubic_function)
    
    q2 = Quadrature1D(order=2)
    assert helper_check_integrand(expected=-13.2, quad=q2, funct=cubic_function)
    
    q3 = Quadrature1D(order=3)
    assert helper_check_integrand(expected=-13.2, quad=q3, funct=cubic_function)

def test_weights_quartic():    
    q2 = Quadrature1D(order=2)
    assert helper_check_integrand(expected=-10.6, quad=q2, funct=quartic_function)
    
    q3 = Quadrature1D(order=3)
    assert helper_check_integrand(expected=-10.6, quad=q3, funct=quartic_function)

###################################################################################################
# Helper functions
###################################################################################################
def helper_check_integrand(expected, quad, funct):
    result = 0.0
    for item in quad:
        result += funct(item[0]) * item[1]
    return result == pytest.approx(expected)

def linear_function(x):
    return 3.1 * x - 4.5

def quadratic_function(x):
    return -6.3 * x*x + 3.1 * x - 4.5

def cubic_function(x):
    return 2.2 * x*x*x - 6.3 * x*x + 3.1 * x - 4.5

def quartic_function(x):
    return 6.5 * x*x*x*x + 2.2 * x*x*x - 6.3 * x*x + 3.1 * x - 4.5