import numpy as np
import scipy
import scipy.signal
import pdb
from astropy.modeling import models,fitting
import numpy.random as rand

def printer():
    print('Hello fitting!')

def legendre2d(d,xo=2,yo=2,xl=None,yl=None):
    """
    Fit a set of 2d Legendre polynomials to a 2D array.
    The aperture is assumed to be +-1 over each dimension.
    NaNs are automatically excluded from the analysis.
    x0 = 2 fits up to quadratic in row axis
    y0 = 3 fits up to cubic in column axis
    If xl and yl are specified, only the polynomials with
    orders xl[i],yl[i] are fitted, this allows for fitting
    only specific polynomial orders.
    """
    #Define fitting algorithm
    fit_p = fitting.LinearLSQFitter()
    #Handle xl,yl
    if xl is not None and yl is not None:
        xo,yo = max(xl),max(yl)
        p_init = models.Legendre2D(xo,yo)
        #Set all variables to fixed
        for l in range(xo+1):
            for m in range(yo+1):
                key = 'c'+str(l)+'_'+str(m)
                p_init.fixed[key] = True
        #p_init.fixed = dict.fromkeys(p_init.fixed.iterkeys(),True)
        #Allow specific orders to vary
        for i in range(len(xl)):
            key = 'c'+str(xl[i])+'_'+str(yl[i])
            p_init.fixed[key] = False
    else:
        # this initializes the model with the number of coefficients
        # given, but with value 0.
        p_init = models.Legendre2D(xo,yo)

    sh = np.shape(d)
    x,y = np.meshgrid(np.linspace(-1,1,sh[1]),\
                      np.linspace(-1,1,sh[0]))
    index = ~np.isnan(d)
    p = fit_p(p_init,x[index],y[index],d[index])
    # p is the model with updated coefficeints to fit the data

    return p(x,y), p.parameters.reshape((yo+1,xo+1))

def fitCylMisalign(d):
    """
    Fit cylindrical misalignment terms to an image
    Piston, tip, tilt, cylindrical sag, and astigmatism
    are fit
    """
    return legendre2d(d,xl=[0,0,1,2,1],yl=[0,1,0,0,1])

def fitConic(d):
    """
    Fit cylindrical misalignment terms to an image
    Piston, tip, tilt, cylindrical sag, and astigmatism
    are fit
    """
    return legendre2d(d,xl=[0,0,1,2,1,2],yl=[0,1,0,0,1,1])

def fitLegendreDistortions(d,xo=2,yo=2,xl=None,yl=None):
    """
    Fit 2D Legendre's to a distortion map as read by 4D.
    If sum of orders is odd, the coefficient needs to be negated.
    """
    #Find and format coefficient arrays
    fit = legendre2d(d,xo=xo,yo=yo,xl=xl,yl=yl)
    az,ax = np.meshgrid(range(xo+1),range(yo+1))
    az = az.flatten()
    ax = ax.flatten()
    coeff = fit[1].flatten()

    #Perform negation
    coeff[(az+ax)%2==1] *= -1.

    return [coeff,ax,az]

def generateDistortion_wRef(xo, yo, ref, N_distortions=1):
    """
    Generate distortion image(s) based on fitting a reference image (ref) with Legendre
    polynomials of order (xo) and (yo). If N_distortions > 1, the distortion images will be returned
    as a 3D array with shape: (N_distortions, ref.shape[0], ref.shape[1])
    """
    # initialize a legendre model of xo, yo order in x and y
    p = models.Legendre2D(xo, yo)
    # get coefficients from reference image
    _, ref_coeffs = legendre2d(ref, xo=xo, yo=yo)
    # get mean and std of reference coefficients
    mu, sigma = np.nanmean(ref_coeffs), np.nanstd(ref_coeffs)
    # get x and y points from reference image
    x, y = np.meshgrid(np.linspace(-1, 1, ref.shape[1]), np.linspace(-1, 1, ref.shape[0]))
    # generate distortion maps
    dist_maps = np.zeros((N_distortions, ref.shape[0], ref.shape[1]))
    for i in range(N_distortions):
        p_coeffs = rand.normal(loc=mu, scale=sigma, size=ref_coeffs.size)
        p.parameters = p_coeffs
        dist_map = p(x,y)
        dist_maps[i] = dist_map#((dist_map-np.nanmin(dist_map))/(dist_map))
    if N_distortions == 1: dist_maps = dist_maps.reshape((ref.shape[0], ref.shape[1]))
    return dist_maps

def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')

def circle(x,y,xc,yc):
    """Fit a circle to a set of x,y coordinates
    Supply with a guess of circle center
    Returns [xc,yc],[rmsRad,rad]
    """
    fun = lambda p: circleMerit(x,y,p[0],p[1])[0]
    res = scipy.optimize.minimize(fun,np.array([xc,yc]),method='Nelder-Mead')
    return res['x'],circleMerit(x,y,res['x'][0],res['x'][1])

def circleMerit(x,y,xo,yo):
    rad = np.sqrt((x-xo)**2+(y-yo)**2)
    mrad = np.mean(rad)
    return np.sqrt(np.mean((rad-mrad)**2)),mrad
