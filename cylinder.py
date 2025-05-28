import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pdb
import imaging.man as man
# import imaging.man as man
import imaging.fitting as fit
import scipy.ndimage as nd
from plotting import scatter3d
import plotting
from imaging.analysis import rms
from scipy.interpolate import griddata

#This module contains routines to fit a cylinder to
#2D metrology data

def printer():
    print('Hello cylinder!')

def cyl(shape, rad, curv=-1, tilt=0, tip=0, roll=0, piston=0, subtractMean=False):
    """Create a cylindrical surface on a 2D array.
    Specify shape of array, and other parameters in
    pixels or radians where appropriate.
    Radius is assumed to be large enough to fill
    provided array.
    Curv is +1 or -1, with +1 indicating a convex
    cylinder from user's point of view (curving negative)"""
    #Construct coordinate grid
    x,y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    x,y = x-np.mean(x),y-np.mean(y)
    #Apply tilt rotation
    x,y = x*np.cos(tilt)+y*np.sin(tilt),-x*np.sin(tilt)+y*np.cos(tilt)
    #Create base cylinder
    cyl = np.sqrt(rad**2-x**2)*curv
    #Apply tip, roll, and piston
    cyl = cyl + piston + x*roll + y*tip
    if subtractMean:
        cyl -= np.nanmean(cyl)

    return cyl

def conic(shape, rad, curv=-1, tilt=0, tip=0, roll=0, piston=0, subtractMean=False):
    """
    Create a conical surface on a 2D array.
    Specify shape of array, and other parameters in
    pixels or radians where appropriate.
    Radius is assumed to be large enough to fill
    provided array.
    rad should be an array a values, with len(rad) = shape[0]
    Curv is +1 or -1, with +1 indicating a convex
    cylinder from user's point of view (curving negative)
    """

    #Construct coordinate grid
    x,y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    x,y = x-np.mean(x),y-np.mean(y)
    #Apply tilt rotation
    x,y = x*np.cos(tilt)+y*np.sin(tilt),-x*np.sin(tilt)+y*np.cos(tilt)
    #Create base cylinder
    rad = rad.reshape(shape[0], 1)
    conic = np.sqrt(rad**2-x**2)*curv
    #Apply tip, roll, and piston
    conic = conic + piston + x*roll + y*tip
    if subtractMean:
        conic -= np.nanmean(conic)

    return conic

def findGuess(d):
    """Find initial guess parameters for cylindrical
    metrology data. Use a quadratic fit in each axis
    to determine curvature sign and radius.
    Assume zero tilt"""
    sh = np.shape(d)
    tilt = 0.
    #Fit x and y slices
    xsl = d[int(sh[0]/2),:]
    ysl = d[:,int(sh[1]/2)]
    xsl,ysl = xsl[np.invert(np.isnan(xsl))],ysl[np.invert(np.isnan(ysl))]
    Nx = np.size(xsl)
    Ny = np.size(ysl)
    xfit = np.polyfit(np.arange(Nx),xsl,1)
    yfit = np.polyfit(np.arange(Ny),ysl,1)
    #Subtract off linear fit
    xsl = xsl - np.polyval(xfit,np.arange(Nx))
    ysl = ysl - np.polyval(yfit,np.arange(Ny))
    #Figure out largest radius of curvature and sign
    Xsag = xsl.max()-xsl.min()
    Ysag = ysl.max()-ysl.min()
    if Xsag > Ysag:
        radius = Nx**2/8./Xsag
        curv = -np.sign(np.mean(np.diff(np.diff(xsl))))
    else:
        radius = Ny**2/8./Ysag
        curv = -np.sign(np.mean(np.diff(np.diff(ysl))))
        tilt = -np.pi/2

    return curv,radius,tilt,np.arctan(yfit[0]),\
                           np.arctan(xfit[0]),\
                           -radius+np.nanmean(d)

def fitCyl(d):
    """Fit a cylinder to the 2D data. NaNs are perfectly fine.
    Supply guess as [curv,rad,tilt,tip,roll,piston]
    """
    guess = findGuess(d)
    fun = lambda p: np.nanmean((d-cyl(np.shape(d),*p))**2)
##    guess = guess[1:]
    # pdb.set_trace()
    res = scipy.optimize.minimize(fun,guess,method='Powell',\
                    options={'disp':True,'ftol':1e-9,'xtol':1e-9})

    return res

def transformCyl(x,y,z,tilt,tip,lateral,piston):
    """Transform x,y,z coordinates for cylindrical fitting"""
    #tilt
    y,z = y*np.cos(tilt)+z*np.sin(tilt), -y*np.sin(tilt)+z*np.cos(tilt)
    #tip
    x,y = x*np.cos(tip)+y*np.sin(tip), -x*np.sin(tip)+y*np.cos(tip)
    #Lateral
    x = x + lateral
    #Piston
    z = z + piston
    return x,y,z

def itransformCyl(x,y,z,tilt,tip,lateral,piston):
    """Transform x,y,z coordinates for cylindrical fitting"""
    #Lateral
    x = x + lateral
    #Piston
    z = z + piston
    #tip
    x,y = x*np.cos(-tip)+y*np.sin(-tip), -x*np.sin(-tip)+y*np.cos(-tip)
    #tilt
    y,z = y*np.cos(-tilt)+z*np.sin(-tilt), -y*np.sin(-tilt)+z*np.cos(-tilt)

    return x,y,z

def meritFn(x,y,z):
    """Merit function for 3d cylindrical fitting"""
    rad = np.sqrt(x**2 + z**2)
    return rms(rad)

def fitCyl3D(d):
    """Fit a cylinder to the 2D data. NaNs are fine. Image is
    first unpacked into x,y,z point data. Use findGuess to
    determine initial guesses for cylindrical axis.
    """
    #Get cylindrical axis guess
    g = findGuess(d)
    #Convert to 3D points
    sh = np.shape(d)
    x,y,z = man.unpackimage(d,xlim=[-sh[1]/2.+.5,sh[1]/2.-.5],\
                            ylim=[-sh[0]/2.+.5,sh[0]/2.-.5])
    xf,yf,zf = man.unpackimage(d,xlim=[-sh[1]/2.+.5,sh[1]/2.-.5],\
                            ylim=[-sh[0]/2.+.5,sh[0]/2.-.5],remove=False)
    #Apply transformations to get points in about the right
    #area
    z = z - np.mean(z)
    x,y = x*np.cos(g[2])+y*np.sin(g[2]), -x*np.sin(g[2])+y*np.cos(g[2])
    y,z = y*np.cos(g[3])+z*np.sin(g[3]), -y*np.sin(g[3])+z*np.cos(g[3])
    x,z = x*np.cos(g[4])+z*np.sin(g[4]), -x*np.sin(g[4])+z*np.cos(g[4])
    z = z + g[0]*g[1]
    #Now apply fit to minimize sum of squares of radii
    fun = lambda p: meritFn(*transformCyl(x,y,z,*p))
    res = scipy.optimize.minimize(fun,[.01,.01,.01,.01],method='Powell',\
                    options={'disp':True})
    x,y,z = transformCyl(x,y,z,*res['x'])
    #Subtract mean cylinder
    rad = np.mean(np.sqrt(x**2+z**2))
    z = z - np.sqrt(rad**2-x**2)
    #Repack into 2D array
    yg,xg = np.meshgrid(np.arange(y.min(),y.max()+1),\
                        np.arange(x.min(),x.max()+1))
    newz = griddata(np.transpose([x,y]),z,\
                    np.transpose([xg,yg]),method='linear')
    pdb.set_trace()
##    xg,yg = np.meshgrid(np.linspace(-sh[1]/2.+.5,sh[1]/2.-.5,sh[1]),\
##                        np.linspace(-sh[0]/2.+.5,sh[0]/2.-.5,sh[0]))
##    pdb.set_trace()
##    newz = griddata(np.transpose([x,y]),z,\
##                    np.transpose([xg,yg]),method='linear')

    return newz


def plot_ROC_and_sag(d, dx, ylim=None):
    """
    Plots the figure at the top, middle, and bottom of a 2D array generated
    using cyl() or conic(). Also plotted is the axial sag of the mirror.
    """
    fig, ax = plt.subplots(1,2)
    xvals_0 = np.linspace(-d.shape[0]/2, d.shape[0]/2, d.shape[0], endpoint=True)*dx
    ax[0].plot(xvals_0, d[0], label='Top of mirror')
    ax[0].plot(xvals_0, d[int(d.shape[0]/2)], label='Middle of mirror')
    ax[0].plot(xvals_0, d[-1], label='Bottom of mirror')
    ax[0].set_xlabel('Azimuthal Dimension (mm)')
    ax[0].set_ylabel('Figure (mm)')
    ax[0].set_title('ROC of Mirror')
    ax[0].legend()
    if ylim:
        ax[0].set_ylim(ylim)
    xvals_1 = np.linspace(d.shape[1]/2, -d.shape[1]/2, d.shape[1], endpoint=True)*dx
    yvals_1 = np.abs(d[:, -1] - d[:, int(d.shape[0]/2)])
    ax[1].plot(xvals_1, yvals_1)
    ax[1].set_xlabel('Axial Dimension (mm)')
    ax[1].set_ylabel('Axial Sag (mm)')
    ax[1].set_title('Axial Sag Along Mirror')
    fig.tight_layout()
    return fig
