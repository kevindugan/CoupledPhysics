from ..Quadrature import Quadrature2D
import pytest

def test_weights_linear():
    q1 = Quadrature2D(order=1)
    assert helper_check_integrand(expected=-18.0, quad=q1, funct=linear_function)
    
    q2 = Quadrature2D(order=2)
    assert helper_check_integrand(expected=-18.0, quad=q2, funct=linear_function)
    
    q3 = Quadrature2D(order=3)
    assert helper_check_integrand(expected=-18.0, quad=q3, funct=linear_function)

def test_weights_quadratic():
    q1 = Quadrature2D(order=1)
    assert helper_check_integrand(expected=-30.2667, quad=q1, funct=quadratic_function)
    
    q2 = Quadrature2D(order=2)
    assert helper_check_integrand(expected=-30.2667, quad=q2, funct=quadratic_function)
    
    q3 = Quadrature2D(order=3)
    assert helper_check_integrand(expected=-30.2667, quad=q3, funct=quadratic_function)

def test_weights_cubic():
    q1 = Quadrature2D(order=1)
    assert helper_check_integrand(expected=-30.2667, quad=q1, funct=cubic_function)
    
    q2 = Quadrature2D(order=2)
    assert helper_check_integrand(expected=-30.2667, quad=q2, funct=cubic_function)
    
    q3 = Quadrature2D(order=3)
    assert helper_check_integrand(expected=-30.2667, quad=q3, funct=cubic_function)

def test_weights_quartic():    
    q2 = Quadrature2D(order=2)
    assert helper_check_integrand(expected=-17.5467, quad=q2, funct=quartic_function)
    
    q3 = Quadrature2D(order=3)
    assert helper_check_integrand(expected=-17.5467, quad=q3, funct=quartic_function)

###################################################################################################
# Helper functions
###################################################################################################
def helper_check_integrand(expected, quad, funct):
    result = 0.0
    for item in quad:
        result += funct(item[0], item[1]) * item[2]
    return result == pytest.approx(expected, rel=5.0e-6)

def linear_function(x,y):
    return 3.1 * x - 5.2 * y - 4.5

def quadratic_function(x,y):
    return -6.3 * x*x + 5.2 * x*y - 2.9 * y*y \
           + 3.1 * x - 5.2 * y - 4.5

def cubic_function(x,y):
    return 2.2 * x*x*x - 6.9 * x*x*y - 4.2 * x*y*y + 3.7 * y*y*y \
           - 6.3 * x*x + 5.2 * x*y - 2.9 * y*y \
           + 3.1 * x - 5.2 * y - 4.5

def quartic_function(x,y):
    return 6.5 * x*x*x*x + 8.3 * x*x*x*y - 7.8 * x*x*y*y - 3.8 * x*y*y*y - 1.6 * y*y*y*y \
           + 2.2 * x*x*x - 6.9 * x*x*y - 4.2 * x*y*y + 3.7 * y*y*y \
           + 3.1 * x - 5.2 * y - 4.5