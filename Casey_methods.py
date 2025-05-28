from numpy import *
import matplotlib.pyplot as plt
import os
import glob
import pickle
import astropy.io.fits as pyfits
from scipy.interpolate import griddata
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb

import utilities.imaging.man as man
import utilities.imaging.stitch as stitch
import utilities.imaging.fitting as fit
import utilities.metrology as met
import fourier

# home_directory = os.getcwd()
# figure_directory = home_directory + '/Figures'
# datafile = '/Users/Casey/Google Drive/LocalResearch/AXRO_C1S04/210201_NewCGH_ActualFullAperture/Data/210201_C1S04_FullMeas_YawAdjust_Meas2.csv'
# old_datafile = '/Users/Casey/Google Drive/LocalResearch/AXRO_C1S04/201009_FullApertureLook/180209_02_C1S04_RefSub.csv'
################################################
## Reducing the data using our metrology functions.

#mirror_data,new_dx = met.readCyl4D(datafile)
#old_data,old_dx = met.readCyl4D(old_datafile)
#old_data = fliplr(man.rotateImage(old_data,rot = pi))

###############################################
def plot_cyl_data_interferometer(cyl_data, cyl_dx, title):
    fig = plt.figure(figsize = (9,9))
    extent = [-shape(cyl_data)[1]*cyl_dx/2,shape(cyl_data)[1]*cyl_dx/2,-shape(cyl_data)[0]*cyl_dx/2,shape(cyl_data)[0]*cyl_dx/2]
    fs = 12

    ax = plt.gca()
    im = ax.imshow(cyl_data,extent = extent,aspect = 'auto',cmap = 'Spectral')
    ax.set_xlabel('Azimuthal Dimension (mm)',fontsize = fs)
    ax.set_ylabel('Axial Dimension (mm)',fontsize = fs)
    ax.set_title(title,fontsize = fs*1.25)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im,cax = cax)
    cbar.set_label('Figure (microns)',fontsize = fs)

    img = cyl_data
    plt.text(0.05,0.13,'RMS: ' + "{:.3}".format(nanstd(img)) + ' um',horizontalalignment = 'left',verticalalignment = 'center',transform = ax.transAxes,fontsize = fs)
    plt.text(0.05,0.09,'PV: ' + "{:.3}".format(nanmax(img) - nanmin(img)) + ' um',horizontalalignment = 'left',verticalalignment = 'center',transform = ax.transAxes, fontsize = fs)

    # os.chdir(figure_directory)
    # plt.savefig(plot_filename)
    # os.chdir(home_directory)
    return fig

#################################################
# Initial plotting of the raw data.
#################################################
#plot_cyl_data_interferometer(mirror_data,new_dx,title = 'C1S04 After Processing -- Full Aperture',plot_filename = '210202_C1S04_FullAperture.png')
#plot_cyl_data_interferometer(old_data,old_dx,title = 'C1S04 Before Processing -- Full Aperture',plot_filename = '201009_C1S04_BeforeProcessing_FullAperture.png')

#################################################
# All related to finding suitable fiducials through subtracting the low frequency figure and then stitching to them.
#################################################
# new_leg_fit = fit.legendre2d(mirror_data,xo=10,yo=10)
# old_leg_fit = fit.legendre2d(old_data,xo=10,yo=10)
#
# new_align_map = mirror_data - new_leg_fit[0]
# old_align_map = old_data - old_leg_fit[0]
#
# plt.ion()
# plt.figure()
# plt.imshow(new_align_map,vmin = -0.5, vmax = 0.5)
# plt.title('New Alignment Map')
#
# plt.figure()
# plt.imshow(old_align_map, vmin = -0.5, vmax = 0.5)
# plt.title('Old Alignment Map')
#
# # Alignment markers stored in a PP file ('210202_AlignmentMarkers.ppx')
#
# with open('C1S04_PickledData.pk1','wb') as f:
#    pickle.dump([mirror_data,new_dx,old_data,old_dx,new_leg_fit,old_leg_fit,new_align_map,old_align_map],f)
#    f.close()
#
# with open('C1S04_PickledData.pk1','rb') as f:
#     [mirror_data,new_dx,old_data,old_dx,new_leg_fit,old_leg_fit,new_align_map,old_align_map,garbage] = pickle.load(f)
#     f.close()

#new_xf = array([461.6,584.4,372.8,452.8,767.0,914.1,1080.5,1290.6,1399.9,1346.2,885.7])
#new_yf = array([993.2,1091.4,467.3,351.6,358.8,67.4,361.8,152.6,300.4,1228.8,1297.0])
#old_xf = array([223.5,279.0,187.7,224.6,384.9,462.5,539.6,648.1,700.2,665.4,425.0])
#old_yf = array([473.6,525.4,217.9,163.2,168.9,33.5,176.0,78.6,148.6,598.5,625.3])

#with open('C1S04_Fiducials.pk1','wb') as f:
#    pickle.dump([new_xf,new_yf,old_xf,old_yf],f)
#    f.close()

#with open('C1S04_Fiducials.pk1','rb') as f:
#    [new_xf,new_yf,old_xf,old_yf] = pickle.load(f)
#    f.close()

#################################################
# Offering more control over the translation alignment, which brings things together nicely in the residual map.

# img1,img2,xf1,yf1,xf2,yf2 = old_data,mirror_data,old_xf,old_yf,new_xf,new_yf
#
# tx,ty,theta,x_mag,y_mag = stitch.matchFiducials_wSeparateMag(xf1,yf1,xf2,yf2)
#
# tx = tx - 8
# ty = ty - 5
#
# x2_wNaNs,y2_wNaNs,z2_wNaNs = man.unpackimage(img2,remove = False,xlim=[0,shape(img2)[1]],\
#                        ylim=[0,shape(img2)[0]])
# #Apply transformations to x,y coords
# x2_wNaNs,y2_wNaNs = stitch.transformCoords_wSeparateMag(x2_wNaNs,y2_wNaNs,tx,ty,theta,x_mag,y_mag)
#
# #Get x,y,z points from reference image
# x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,shape(img1)[1]],\
#                        ylim=[0,shape(img1)[0]])
#
# #Interpolate stitched image onto expanded image grid
# newimg = griddata((x2_wNaNs,y2_wNaNs),z2_wNaNs,(x1,y1),method='linear')
# print('Interpolation ok')
# newimg = newimg.reshape(shape(img1))
#
# #Images should now be in the same reference frame
# #Time to apply tip/tilt/piston to minimize RMS
# aligned_new_data = stitch.matchPistonTipTilt(img1,newimg)
#
# with open('C1S04_PickledData.pk1','wb') as f:
#    pickle.dump([mirror_data,new_dx,old_data,old_dx,new_leg_fit,old_leg_fit,new_align_map,old_align_map,aligned_new_data],f)
#    f.close()

##################################################
## All related to finding suitable fiducials through subtracting the low frequency figure and then stitching to them.
##################################################
# with open('C1S04_PickledData.pk1','rb') as f:
#     [mirror_data,new_dx,old_data,old_dx,new_leg_fit,old_leg_fit,new_align_map,old_align_map,aligned_new_data] = pickle.load(f)
#     f.close()
#
# plot_cyl_data_interferometer(aligned_new_data,old_dx,title = 'C1S04 After Processing -- Aligned',plot_filename = '210208_C1S04_NewData_OldAligned.png')
# plot_cyl_data_interferometer(old_data - aligned_new_data,old_dx,title = 'C1S04 -- Figure Change (old - new)',plot_filename = '210208_C1S04_FigureChange.png')
#
# align_check = fit.legendre2d(old_data - aligned_new_data,xo=10,yo=10)
# diff_img = old_data - aligned_new_data - align_check[0]
# plt.figure()
# plt.imshow(diff_img,vmin = -0.25, vmax = 0.25)
# plt.xlabel('Azimuthal Dimension (pixels)')
# plt.ylabel('Axial Dimension (pixels)')
# plt.title('Difference Image, After Legendre Subtraction Through Order 10')
# plt.savefig('210209_C1S04_AlignmentCrossCheck_DifferenceMinusLegendres.png')
