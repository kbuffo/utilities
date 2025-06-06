import numpy as np
import matplotlib.pyplot as plt
import utilities.imaging.man as man
import utilities.imaging.fitting as fit
import scipy.ndimage as nd
from linecache import getline
import astropy.io.fits as pyfits
import pdb
import h5py
import csv

### 2025 update

def printer():
    print('Hello metrology!')

def readCylScript(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print('Wedge factor is ' + str(wedge))
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *wave
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readCyl4D(fn,rotate=np.linspace(.75,1.5,50),interp=None, fliplr=True):
    """
    Load in data from 4D measurement of cylindrical mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    fliplr will flip the left and right sides of an image since taking a cylindrical
    measurement with the CGH will flip it across the axial axis when viewing in 4D
    """
    #Get xpix value in mm
    l = getline(fn,9)
    wedge = float(getline(fn, 8).split()[-1])
    if wedge != 0.5:
        print('Wedge != 0.5, wedge = {}'.format(wedge))
    dx = float(l.split()[1])*1000.

    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    print('array size before manipulation:', d.shape)
    d = man.stripnans(d)
    d = d *.6328 * wedge
    d = d - np.nanmean(d)
    print('array size after striping nans:', d.shape)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]
    print('array size after removing cyl misalign terms:', d.shape)

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))
    print('array size after rotation:', d.shape)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)
    print('final array size:', d.shape)
    if fliplr: d = np.fliplr(d) # we fliplr to compensate for the CGH flipping the measurement
    return d,dx

def readCyl4D_h5(h5_file, rotate=np.linspace(.75,1.5,50), fliplr=True):
    f = h5py.File(h5_file, 'r')
    meas = f['measurement0']

    #################
    # Getting the attributes of the data directly from the .h5 file
    wedge = meas['genraw'].attrs['wedge']
    if wedge != 0.5:
        print('Error: wedge != 0.5, wedge = {}'.format(wedge))
        return None
    height_unit = meas['genraw'].attrs['height_units']
    wave = meas['genraw'].attrs['wavelength']
    xpix = meas['genraw'].attrs['xpix']

    #################
    # Processing the data as though it's a cylinder.
    # Apply the wedge factor.
    raw_data = np.array(meas['genraw']['data'])  #* wedge
    # Removing the absurdly large value defaulted to for bad data and replacing it with a NaN.
    raw_data[raw_data > 1e10] = np.NaN
    # Then stripping that bad data flagged as nans from the perimeter.
    data = man.stripnans(raw_data)
    # Set the average surface to zero (i.e., remove piston)
    data -= np.nanmean(data)
    # Remove cylindrical misalignment.
    data = data - fit.fitCylMisalign(data)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(data,a,order=1,cval=np.nan)))) for a in rotate]
        data = man.stripnans(\
            nd.rotate(data,rotate[np.argmin(b)],order=1,cval=np.nan))

    # Apply unit transformation converting data to microns.
    height_unit = height_unit.decode('UTF-8')
    if height_unit == 'wv':
        wavelength = float(wave[:-3])
        data *= wavelength/1000
    if height_unit == 'nm':
        data /= 1000

    # Apply unit transformation converting pixel size to mm.
    xpix = xpix.decode('UTF-8')
    # print('xpix:', xpix)
    # print('xpix type:', type(xpix))
    # print('xpix.find(''):', xpix.find(' '))
    # print(xpix[3])
    pix_unit = xpix[xpix.find(' ') + 1:]
    pix_num = float(xpix[:xpix.find(' ')])

    if pix_unit == 'inch':
        pix_num *= 25.4
    if fliplr: 
        data = np.fliplr(data) # we fliplr to compensate for the CGH flipping the measurement
    return data, pix_num

def readConic4D(fn,rotate=None,interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.

    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    conic_fit = fit.fitConic(d)
    d = d - conic_fit[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx,conic_fit[1]

def readFlatScript(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print('Wedge factor is ' + str(wedge))
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readFlat4D(fn,interp=None,printLength=True):
    """
    Load in data from 4D measurement of flat mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    wedge = float(getline(fn, 8).split()[-1])
    if wedge != 0.5:
        print('Wedge != 0.5, wedge = {}'.format(wedge))
        return None
    dx = float(l.split()[1])*1000.
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328 * wedge
    d = d - np.nanmean(d)
    # Remove tip from data
    d -= fit.legendre2d(d, xo=0, yo=1)[0]
    # Remove tilt from data
    d -= fit.legendre2d(d, xo=1, yo=0)[0]

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)
    if printLength:
        print('The optic for {} is {:.2f} in long.'.format(fn, (d.shape[0]*dx)/25.4))
    return d,dx

def readFlat4D_h5(h5_file, removeTipTilt=True, applyWedge=False):
    f = h5py.File(h5_file, 'r')
    meas = f['measurement0']
    # print('meas keys:', meas.keys())
    # print('genraw keys:', meas['genraw'].keys())
    # print('analyzed keys:', meas['analyzed'].keys())
    #################
    # Getting the attributes of the data directly from the .h5 file
    wedge = meas['genraw'].attrs['wedge']
    if wedge != 0.5:
        print('Error: wedge != 0.5, wedge = {}'.format(wedge))
        return None
    height_unit = meas['genraw'].attrs['height_units']
    wave = meas['genraw'].attrs['wavelength']
    xpix = meas['genraw'].attrs['xpix']
    #################
    # Processing the data as though it's a cylinder.
    raw_data = np.array(meas['genraw']['data'])
    # Apply the wedge factor.
    #if applyWedge:
    #    raw_data *= wedge
    # Removing the absurdly large value defaulted to for bad data and replacing it with a NaN.
    raw_data[raw_data > 1e10] = np.NaN
    # Then stripping that bad data flagged as nans from the perimeter.
    data = man.stripnans(raw_data)
    # Set the average surface to zero (i.e., remove piston)
    data -= np.nanmean(data)
    if removeTipTilt:
        # Remove tip from data
        data -= fit.legendre2d(data, xo=0, yo=1)[0]
        # Remove tilt from data
        data -= fit.legendre2d(data, xo=1, yo=0)[0]
    # Apply unit transformation converting data to microns.
    height_unit = height_unit.decode('UTF-8')
    # print('height unit:', height_unit)
    if height_unit == 'wv':
        wavelength = float(wave[:-3])
        data *= wavelength/1000
    if height_unit == 'nm':
        data /= 1000
    # Apply unit transformation converting pixel size to mm.
    xpix = xpix.decode('UTF-8')
    pix_unit = xpix[xpix.find(' ') + 1:]
    pix_num = float(xpix[:xpix.find(' ')])
    if pix_unit == 'inch':
        pix_num *= 25.4
    return data, pix_num

def readQCstats_csv(csv_file):
    """
    Reads a stats csv file from doing a QC measurement in 4Sight. N is the 
    number of averaged measurements in the QC measurement. Units are specified 
    in the file provided.
    Returns:
    PVr: 1D array of PVr values for each averaged measurement and has len(N)
    RMS: 1D array of RMS values for each averaged measurement and has len(N)
    delta_PVr: 1D array of PVr values for each delta averaged measurement and has len(N)
    delta_RMS: 1D array of RMS values for each delta averaged measurement adn has len(N)
    """
    PVr, RMS, delta_PVr, delta_RMS = [], [], [], []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        line_num = 0
        meas_num = 1
        for line in csv_reader:
            if line and line[0] == str(meas_num):
                PVr.append(float(line[1]))
                RMS.append(float(line[2]))
                delta_PVr.append(float(line[3]))
                delta_RMS.append(float(line[4]))
                meas_num = int(meas_num) + 1
        return np.array(PVr), np.array(RMS), np.array(delta_PVr), np.array(delta_RMS)
                
def readQCparams_csv(csv_file):
    """
    Reads a parameters file from doing a QC measurement in 4Sight. N is the 
    number of averaged measurements in the QC measurement. Units
    are specified in the file provided.
    Returns:
    PVr_uncal_acc: The uncalibrated PVr accuracy, the PVr of averaging N averaged measurements
    RMS_uncal_acc: The uncalibrated RMS accuracy, the RMS of averaging N averaged measurements
    PVr_rep: The PVr repeatability, the standard deviation of N PVr values
    RMS_rep: The RMS repeatability, the standard deviation of N RMS values
    PVr_prec: The PVr precision, the mean of the delta PVr values
    RMS_prec: The RMS precision, the mean of the delta RMS values
    """
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        line_num = 0
        for line in csv_reader:
            if len(line) == 6:
                if line[1] == 'PVr Uncalibrated Accuracy':
                    PVr_uncal_acc = float(line[3])
                elif line[1] == 'RMS Uncalibrated Accuracy':
                    RMS_uncal_acc = float(line[3])
                elif line[1] == 'PVr Repeatibility':
                    PVr_rep = float(line[3])
                elif line[1] == 'RMS Repeatibility':
                    RMS_rep = float(line[3])
                elif line[1] == 'PVr Precision':
                    PVr_prec = float(line[3])
                elif line[1] == 'RMS Precision':
                    RMS_prec = float(line[3])
                else:
                    continue
            line_num += 1
        return PVr_uncal_acc, RMS_uncal_acc, PVr_rep, RMS_rep, PVr_prec, RMS_prec

def write4DFits(filename,img,dx,dx2=None):
    """
    Write processed 4D data into a FITS file.
    Axial pixel size is given by dx.
    Azimuthal pixel size is given by dx2 - default to none
    """
    hdr = pyfits.Header()
    hdr['DX'] = dx
    hdu = pyfits.PrimaryHDU(data=img,header=hdr)
    hdu.writeto(filename,clobber=True)
    return

def read4DFits(filename):
    """
    Write FITS file of processed 4D data.
    Returns img,dx in list
    """
    dx = pyfits.getval(filename,'DX')
    img = pyfits.getdata(filename)
    return [img,dx]

def readCylWFS(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from WFS measurement of cylindrical mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    # Negate to make bump positive and rotate to be consistent with looking at the part beamside.
    d = -np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d

def readConicWFS(fn,interp=None):
    """
    Load in data from WFS measurement of cylindrical mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    Returns the data with best fit conic removed, as well as the
    coefficients in the conic fit.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = d - np.nanmean(d)

    # Negate to make bump positive and rotate to be consistent with looking at the part beamside.
    d = -np.fliplr(d)

    #Remove cylindrical misalignment terms
    conic_fit = fit.fitConic(d)
    d = d - conic_fit[0]

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,conic_fit[1]

def readFlatWFS(fn,interp=None):
    """
    Load in data from WFS measurement of flat mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)
    d = -d

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d

#Read in Zygo ASCII file
def readzygo(filename):
    #Open file
    f = open(filename,'r')

    #Read third line to get intensity shape
    for i in range(3):
        l = f.readline()
    l = l.split(' ')
    iwidth = int(l[2])
    iheight = int(l[3])

    #Read fourth line to get phase shape
    l = f.readline()
    l = l.split(' ')
    pwidth = int(l[2])
    pheight = int(l[3])

    #Read eighth line to get scale factors
    for i in range(4):
        l = f.readline()
    l = l.split(' ')
    scale = float(l[1])
    wave = float(l[2])
    o = float(l[4])
    latscale = float(l[6])

    #Read eleventh line to get phase resolution
    f.readline()
    f.readline()
    l = f.readline()
    l = l.split(' ')
    phaseres = l[0]
    if phaseres == 0:
        phaseres = 4096
    else:
        phaseres = 32768

    #Read through to first '#' to signify intensity
    while (l[0]!='#'):
        l = f.readline()

    #Read intensity array
    #If no intensity, l will be '#' below
    l = f.readline()
    while (l[0]!='#'):
        #Convert to array of floats
        l = np.array(l.split(' '))
        l = l[:-1].astype('float')
        #Merge into intensity array
        try:
            intensity = np.concatenate((intensity,l))
        except:
            intensity = l
        #Read next line
        l = f.readline()

    #Reshape into proper array
    try:
        intensity = np.reshape(intensity,(iheight,iwidth))
    except:
        intensity = np.nan

    #Read phase array
    l = f.readline()
    while (l!=''):
        #Convert to array of floats
        l = np.array(l.split(' '))
        l = l[:-1].astype('float')
        #Merge into intensity array
        try:
            phase = np.concatenate((phase,l))
        except:
            phase = l
        #Read next line
        l = f.readline()

    phase = np.reshape(phase,(pheight,pwidth))
    phase[np.where(phase==phase.max())] = np.nan
    phase = phase*scale*o*wave/phaseres
    f.close()
    print(wave, scale, o, phaseres)

    return intensity, phase, latscale

#Convert Zygo ASCII to easily readable ASCII format
def convertzygo(filename):
    #read in zygo data
    intensity,phase,latscale = readzygo(filename)

    np.savetxt(filename.split('.')[0]+'.txt',phase,header='Lat scale: '+\
            str(latscale)+'\n'+'Units: meters')

def make_extent(data,dx):
    return [-float(np.shape(data)[1])/2*dx,float(np.shape(data)[1])/2*dx,-float(np.shape(data)[0])/2*dx,float(np.shape(data)[0])/2*dx]
