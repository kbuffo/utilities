import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

def printer():
    print('Hello chebFit!')

def chebFit(d,xcoeff,ycoeff):
    """Perform a fit to 2D data with 2D Chebyshev polynomials.
    """
