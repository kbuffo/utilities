import numpy as np
import utilities.imaging.man as man
import utilities.imaging.analysis as alsis
from utilities.imaging.zernikemod import fitimg

def make_radial_aperture_image(d_input, dx, radius, center_coords=None, stripnans=False, subtractmean=False):
    """
    Returns a circular image of a specified radius centered around ceter_coords.
    radius should be specified in physical units (same units as dx which is in units/pix).
    center_coords should be specified as (d_input row, d_input column). Default is to use the center of the array.
    """
    d = np.copy(d_input)
    if center_coords is None:
        center_coords = [int(d.shape[0]/2), int(d.shape[1]/2)]
    x, y = np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0]))
    radial_positions = np.sqrt((x-center_coords[1])**2 + (y-center_coords[0])**2) * dx
    valid_locations = radial_positions < radius
    d[~valid_locations] = np.nan
    if stripnans:
        d = man.stripnans(d)
    if subtractmean:
        d -= np.nanmean(d)
    return d

def fit_defocus_to_img(d):
    """
    Takes in a 2D array d and returns a Zernike defocus (power) fit to the data and the associated defocus coefficient.
    d needs to be equal size in x and y.
    """
    fit_coeffs, defocus_fit = fitimg(d, N=None, r=np.array([2]), m=np.array([0]))
    defocus_coeff = fit_coeffs[0][0]
    return defocus_fit, defocus_coeff

def calc_ROC(d, dx, D=None, pv=None):
    """
    Calculates the radius of curvature (ROC) for defocus fit data array.
    
    d: 2D data array, square in shape (in mircrons)
    dx: pixel size of array (mm/pix)
    D: diameter of data (in meters)
    pv: peak-to-valley of the defocus fit array. If None, it will be calculated from d directly. Thus, d should
        be the Zernike defocus fit array
    Returns: the ROC (in meters)
    """
    if pv is None:
        pv = alsis.ptov(d)
    if D is None:
        D = np.shape(d)[0] * dx * 1e-3
    ROC = D**2 / (8*pv*1e-6)
    return ROC

def calc_integrated_stress_from_ROC(roc, h):
    """
    Calculates the integrated stress (in Pa*m) on a Si(100) flat substrate.
    roc: radius of curvature in meters (use calc_ROC()).
    h: substrate thickness in um.
    B_100 is the biaxial modulus for Si(100).
    """
    B_100 = 1.794e11 # Pa
    h *= 1.e-6 # m
    int_stress = (h**2 * B_100) / (6 * roc)
    return int_stress
    