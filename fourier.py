import numpy as np
import pdb
from scipy.interpolate import griddata
from imaging.man import stripnans,nearestNaN
import imaging.analysis as alsis
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#This module contains Fourier analysis routine

def printer():
    print('Hello fourier!')

def components(d,win=np.hanning):
    """Want to return Fourier components with optional window
    Application note: These components are dependent on sampling!
    This means you can *not* interpolate these components onto other
    frequency grids!
    """
    #Handle window
    if win != 1:
        if np.size(np.shape(d)) == 1: # if the data array is 1D
            window = win(np.size(d))/np.sqrt(np.mean(win(np.size(d))**2))
        else:
            win1 = win(np.shape(d)[0])
            win2 = win(np.shape(d)[1])
            window = np.outer(win1,win2)
            window = window/np.sqrt(np.mean(window**2))
    else: window = 1

    #Compute Fourier components
    return np.fft.fftn(d*window)/np.size(d)

def continuousComponents(d,dx,win=np.hanning):
    """Want to return Fourier components with optional window
    Divide by frequency interval to convert to continuous FFT
    These components can be safely interpolated onto other frequency
    grids. Multiply by new frequency interval to get to numpy format
    FFT. Frequency units *must* be the same in each case.
    """
    #Handle window
    if win != 1:
        if np.size(np.shape(d)) == 1:
            window = win(np.size(d))/np.sqrt(np.mean(win(np.size(d))**2))
        else:
            win1 = win(np.shape(d)[0])
            win2 = win(np.shape(d)[1])
            window = np.outer(win1,win2)
            window = window/np.sqrt(np.mean(window**2))

    #Compute Fourier components
    return np.fft.fftn(d*window)*dx

def newFreq(f,p,nf):
    """
    Interpolate a power spectrum onto a new frequency grid.
    """
    return griddata(f,p,nf,method=method)

def freqgrid(d,dx=1.):
    """Return a frequency grid to match FFT components
    """
    freqx = np.fft.fftfreq(np.shape(d)[1],d=dx)
    freqy = np.fft.fftfreq(np.shape(d)[0],d=dx)
    freqx,freqy = np.meshgrid(freqx,freqy)
    return freqx,freqy

def ellipsoidalHighFrequencyCutoff(d,fxmax,fymax,dx=1.,win=np.hanning):
    """A simple low-pass filter with a high frequency cutoff.
    The cutoff boundary is an ellipsoid in frequency space.
    All frequency components with (fx/fxmax)**2+(fy/fymax)**2 > 1.
    are eliminated.
    fxmax refers to the second index, fymax refers to the first index
    This is consistent with indices in imshow
    """
    #FFT components in numpy format
    fftcomp = components(d,win=win)*np.size(d)

    #Get frequencies
    freqx,freqy = freqgrid(d,dx=dx)

    #Get indices of frequencies violating cutoff
    ind = (freqx/fxmax)**2+(freqy/fymax)**2 > 1.
    fftcomp[ind] = 0.

    #Invert the FFT and return the filtered image
    return fft.ifftn(fftcomp)

def meanPSD(d0,win=np.hanning,dx=1.,axis=0,irregular=False,returnInd=False,minpx=10):
    """Return the 1D PSD averaged over a surface.
    Axis indicates the axis over which to FFT
    If irregular is True, each slice will be stripped
    and then the power spectra
    interpolated to common frequency grid
    Presume image has already been interpolated internally
    If returnInd is true, return array of power spectra
    Ignores slices with less than minpx non-nans
    """
    #Handle which axis is transformed
    if axis==0:
        d0 = np.transpose(d0) # each row of the 2d array is an axial strip
    #Create list of slices
    if irregular is True: # remove the nans from each row in the array
        d0 = [stripnans(di) for di in d0]
    #Create power spectra from each slice
    pows = [realPSD(s,win=win,dx=dx,minpx=minpx) for s in d0 \
            if np.sum(~np.isnan(s)) >= minpx]
    #Interpolate onto common frequency grid of shortest slice
    if irregular is True:
        #Determine smallest frequency grid
        ln = [len(s[0]) for s in pows]
        freq = pows[np.argmin(ln)][0]
        #Interpolate
        pp = [griddata(p[0],p[1],freq) for p in pows]
    else:
        pp = [p[1] for p in pows]
        freq = pows[0][0]
    #Average
    pa = np.mean(pp,axis=0)
    if returnInd is True:
        return freq,pp
    return freq,pa

def medianPSD(d0,win=np.hanning,dx=1.,axis=0,nans=False):
    """Return the 1D PSD "medianed" over a surface.
    Axis indicates the axis over which to FFT
    If nans is True, each slice will be stripped,
    internally interpolated, and then the power spectra
    interpolated to common frequency grid"""
    d = stripnans(d0)
    if win != 1:
        win = win(np.shape(d)[axis])/\
              np.sqrt(np.mean(win(np.shape(d)[axis])**2))
        win = np.repeat(win,np.shape(d)[axis-1])
        win = np.reshape(win,(np.shape(d)[axis],np.shape(d)[axis-1]))
        if axis == 1:
            win = np.transpose(win)
    c = np.abs(np.fft.fft(d*win,axis=axis)/np.shape(d)[axis])**2
    c = np.median(c,axis=axis-1)
    f = np.fft.fftfreq(np.size(c),d=dx)
    f = f[:np.size(c)/2]
    c = c[:np.size(c)/2]
    c[1:] = 2*c[1:]
    return f,c

def realPSD(d0,win=np.hanning,dx=1.,axis=None,nans=False,minpx=10):
    """This function returns the PSD of a real function
    Gets rid of zero frequency and puts all power in positive frequencies
    Returns only positive frequencies
    """
    if nans is True:
        d = stripnans(d0)
    else:
        d = d0
    if len(d) < minpx:
        return np.nan
    #Get Fourier components
    c = components(d,win=win)
    #Handle collapsing to 1D PSD if axis keyword is set
    if axis==0:
        c = c[:,0]
    elif axis==1:
        c = c[0,:]

    #Reform into PSD
    if np.size(np.shape(c)) == 2:
        f = [np.fft.fftfreq(np.shape(c)[0],d=dx)[:np.shape(c)[0]/2],\
                   np.fft.fftfreq(np.shape(c)[1],d=dx)[:np.shape(c)[1]/2]]
        c = c[:np.shape(c)[0]/2,:np.shape(c)[1]/2]
        c[0,0] = 0.
        #Handle normalization
        c = 2*c
        c[0,:] = c[0,:]/np.sqrt(2.)
        c[:,0] = c[:,0]/np.sqrt(2.)

    elif np.size(np.shape(c)) == 1:
        f = np.fft.fftfreq(np.size(c),d=dx)
        f = f[:int(np.size(c)/2)]
        c = c[:int(np.size(c)/2)]
        c[0] = 0.
        c = c*np.sqrt(2.)

    return f[1:],np.abs(c[1:])**2

def computeFreqBand(f,p,f1,f2,df,method='linear'):
    """
    Compute the power in the PSD between f1 and f2.
    f and p should be as returned by realPSD or meanPSD
    Interpolate between f1 and f2 with size df
    Then use numerical integration
    """
    newf = np.linspace(f1,f2,int((f2-f1)/df+1))
    try:
        newp = griddata(f,p/f[0],newf,method=method)
    except:
        pdb.set_trace()
    return np.sqrt(simps(newp,x=newf))

def fftComputeFreqBand(d,f1,f2,df,dx=1.,win=np.hanning,nans=False,minpx=10,\
                       method='linear'):
    """
    Wrapper to take the FFT and immediately return the
    power between f1 and f2 of a slice
    If slice length is < 10, return nan
    """
    if np.sum(~np.isnan(d)) < minpx:
        return np.nan
    f,p = realPSD(d,dx=dx,win=win,nans=nans)
    return computeFreqBand(f,p,f1,f2,df,method=method)

def psdScan(d,f1,f2,df,N,axis=0,dx=1.,win=np.hanning,nans=False,minpx=10):
    """
    Take a running slice of length N and compute band limited
    power over the entire image. Resulting power array will be
    of shape (S1-N,S2) if axis is 0
    axis is which axis to FFT over
    """
    if axis == 0:
        d = np.transpose(d)
    sh = np.shape(d)
    m = np.array([[fftComputeFreqBand(di[i:i+N],f1,f2,df,dx=dx,win=win,nans=nans,minpx=minpx) \
      for i in range(sh[1]-N)] for di in d])
    if axis == 0:
        m = np.transpose(m)
    return m

def PSD_stack(d, dx=1, norm=False):
    """
    Returns an array of shape (I, J, K, L) where i indexes the distortion number,
    j indexes the direction of the PSD (j=0 for y and j=1 for x), k indexes either the frequency
    or the coefficients (k=0 for freq values, k=1 for coeff values), and L indexes the value in PSD.
    """
    if d.ndim == 2:
        d = d.reshape(1, d.shape[0], d.shape[1])
    PSD_stack_array = []
    for i in range(d.shape[0]):
        freq_y, pa_y = meanPSD(d[i], dx=dx, axis=0) # compute PSD in x and y
        freq_x, pa_x = meanPSD(d[i], dx=dx, axis=1)
        # freq_y, pa_y = realPSD(d[i], dx=dx, axis=0) # compute PSD in x and y
        # freq_x, pa_x = realPSD(d[i], dx=dx, axis=1)
        rms = alsis.rms(d[i]) # get rms value of array

        if norm:
            sum_rms_y = np.sqrt(np.sum(pa_y))
            sum_rms_x = np.sqrt(np.sum(pa_x))
            pa_y = pa_y * (rms/sum_rms_y)**2
            pa_x = pa_x * (rms/sum_rms_x)**2
            # pa_y = pa_y / (2*np.pi*freq_y*(freq_y[1]-freq_y[0]))
            # pa_x = pa_x / (2*np.pi*freq_x*(freq_x[1]-freq_x[0]))

        psd_y = np.stack([freq_y, pa_y], axis=0)
        psd_x = np.stack([freq_x, pa_x], axis=0)
        dist_psd_stack = np.stack([psd_y, psd_x], axis=0)
        PSD_stack_array.append(dist_psd_stack)
    PSD_stack_array = np.stack(PSD_stack_array, axis=0)
    return PSD_stack_array

def compute_PSD_stack_RMS(psd_stack, map_no, df, f1=None, f2=None, method='linear',
                            printout=False):
    """
    Computes the RMS over PSD that was obtained using PSD_stack(), taking into account
    both the axial and azimuthal PSDs. If f1 and f2 are specified, return the RMS of the
    given frequency band. If they are None, compute the RMS over the entire frequency band of the PSD.
    """
    psd_array = psd_stack[map_no]
    axial_f, axial_c = psd_array[0, 0, :], psd_array[0, 1, :]
    az_f, az_c = psd_array[1, 0, :], psd_array[1, 1, :]
    if f1 is None:
        axial_f1 = axial_f[0]
        az_f1 = az_f[0]
    else:
        axial_f1 = f1
        az_f1 = f1
    if f2 is None:
        axial_f2 = axial_f[-1]
        az_f2 = az_f[-1]
    else:
        axial_f2 = f2
        az_f2 = f2
    new_axial_f = np.linspace(axial_f1, axial_f2, int((axial_f2-axial_f1)/df+1))
    new_az_f = np.linspace(az_f1, az_f2, int((az_f2-az_f1)/df+1))
    try:
        new_axial_p = griddata(axial_f, axial_c/axial_f[0], new_axial_f, method=method) # axial_f[0]
        new_az_p = griddata(az_f, az_c/az_f[0], new_az_f, method=method) # az_f[0]
    except:
        print('Error during griddata.')
        pdb.set_trace()
    axial_rms = np.sqrt(simps(new_axial_p, x=new_axial_f))
    az_rms = np.sqrt(simps(new_axial_p, x=new_axial_f))#np.sqrt(simps(new_az_p, x=new_az_f))
    rms = np.sqrt(axial_rms**2+az_rms**2)
    if printout:
        print('axial rms: {:.3f}'.format(axial_rms))
        print('azimuthal rms: {:.3f}'.format(az_rms))
        print('total rms: {:.3f}'.format(rms))
    return rms

def PSD_plot(PSD_stacks, dist_num, PSD_axis,
            labels=None, dtype='um', dist_num_label=None,
            figsize=None, title_fontsize=14, ax_fontsize=12,
            title=None, xlabel=None, ylabel=None, colors=None, linestyles=None,
            freq_limit=None, freq_limit_label=None, freq_line_top_end=1,
            includeLegend=True, legendCoords=(1.65, 0.5), legendCols=1, legendLoc='right',
            xlims=None, ylims=None, include_RMS=True):
    N_lines = len(PSD_stacks)
    if not labels:
        labels = [''] * len(PSD_stacks)
    if not colors:
        colors = list(mcolors.TABLEAU_COLORS)[:N_lines]
    if not linestyles:
        linestyles = ['solid'] * N_lines
    if ylabel is None:
        if dtype == 'um':
            ylabel = r'Power $\left({\mu m}^2\;mm\right)$'
        if dtype == 'arcsec':
            ylabel = r'Power $\left({\mathrm{arcsec}}^2\;\mathrm{mm}\right)$'
    if xlabel is None:
        xlabel = 'Frequency (1/mm)'
    fig, ax = plt.subplots()
    for i in range(N_lines):
        PSD_array = PSD_stacks[i]
        xvals = PSD_array[dist_num, PSD_axis, 0]
        yvals = PSD_array[dist_num, PSD_axis, 1]
        rmsval = compute_PSD_stack_RMS(PSD_array, dist_num, 1e-5)
        if include_RMS:
            rmsLabel = 'RMS = {:.2f} {}'.format(rmsval, dtype)
            labels = [label+'\n' for label in labels]
        else:
            rmsLabel = ''
        ax.plot(xvals, yvals, color=colors[i], marker='.',
                linestyle=linestyles[i], label=labels[i]+rmsLabel)
    ax.set_xlabel(xlabel, fontsize=ax_fontsize)
    ax.set_ylabel(ylabel, fontsize=ax_fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    if figsize:
        fig.set_size_inches(figsize[0], figsize[1])
    if freq_limit is not None:
        ax.axvline(freq_limit, ymax=freq_line_top_end, color='black',
                linestyle='dashed', label=freq_limit_label,
                    zorder=0)
    if PSD_axis == 0:
        PSD_direction = 'Axial'
    if PSD_axis == 1:
        PSD_direction = 'Azimuthal'
    if dtype == 'um':
        space_tag = 'Figure Space'
    if dtype == 'arcsec':
        space_tag = 'Slope Space'
    if dist_num_label is None:
        dist_num_label = dist_num
    if xlims is not None:
        print('xlims:', xlims)
        ax.set_xlim(xmin=xlims[0], xmax=xlims[1])
    if ylims is not None:
        ax.set_ylim(ymin=ylims[0], ymax=ylims[1])
    if title is None:
        title = '{} Power Spectral Density (PSD) of\nDistortion: {} -- {}'\
                    .format(PSD_direction, dist_num_label, space_tag)
    ax.set_title(title, fontsize=title_fontsize)
    if includeLegend:
        ax.legend(ncol=legendCols, bbox_to_anchor=legendCoords, loc=legendLoc,
                    fontsize=ax_fontsize, framealpha=0.)
    return fig


def lowpass(d,dx,fcut):
    """Apply a low pass filter to a 1 or 2 dimensional array.
    Supply the bin size and the cutoff frequency in the same units.
    """
    #Get shape of array
    sh = np.shape(d)
    #Take FFT and form frequency arrays
    f = np.fft.fftn(d)
    if np.size(np.shape(d)) > 1:
        fx = np.fft.fftfreq(sh[0],d=dx)
        fy = np.fft.fftfreq(sh[1],d=dx)
        fa = np.meshgrid(fy,fx)
        fr = np.sqrt(fa[0]**2+fa[1]**2)
    else:
        fr = np.fft.fftfreq(sh[0],d=dx)
    #Apply cutoff
    f[fr>fcut] = 0.
    #Inverse FFT
    filtered = np.fft.ifftn(f)
    return filtered

def randomizePh(d):
    """Create a randomized phase array that maintains a real
    inverse Fourier transform. This requires that F(-w1,-w2)=F*(w1,w2)
    """
    #Initialize random phase array
    sh = np.shape(d)
    ph = np.zeros(sh,dtype='complex')+1.

    #Handle 1D case first
    if np.size(sh) == 1:
        if np.size(d) % 2 == 0:
            ph[1:sh[0]/2] = np.exp(1j*np.random.rand(sh[0]/2-1)*2*np.pi)
            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2]))
        else:
            ph[1:sh[0]/2+1] = np.exp(1j*np.random.rand(sh[0]/2)*2*np.pi)
            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2+1]))
    else:
        #Handle zero frequency column/rows
        ph[:,0] = randomizePh(ph[:,0])
        ph[0,:] = randomizePh(ph[0,:])
        #Create quadrant
        if sh[0] % 2 == 0 and sh[1] % 2 == 0:
            #Handle intermediate Nyquist
            ph[sh[0]/2,:] = randomizePh(ph[sh[0]/2,:])
            ph[:,sh[1]/2] = randomizePh(ph[:,sh[1]/2])
            #Form quadrant
            ph[1:sh[0]/2,1:sh[1]/2] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,1:sh[1]/2])))
            ph[1:sh[0]/2,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,sh[1]/2+1:])))
        elif sh[0] % 2 == 0 and sh[1] % 2 == 1:
            #Handle intermediate Nyquist
            ph[sh[0]/2,:] = randomizePh(ph[sh[0]/2,:])
            #Form quadrant
            ph[1:sh[0]/2,1:sh[1]/2+1] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,1:sh[1]/2+1])))
            ph[1:sh[0]/2,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2-1,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2,sh[1]/2+1:])))
        elif sh[0] % 2 == 1 and sh[1] % 2 == 0:
            #Handle intermediate Nyquist
            ph[:,sh[1]/2] = randomizePh(ph[:,sh[1]/2])
            #Form quadrant
            ph[1:sh[0]/2+1,1:sh[1]/2] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,1:sh[1]/2])))
            ph[1:sh[0]/2+1,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2-1)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,sh[1]/2:])))
        else:
            #Form quadrant
            ph[1:sh[0]/2+1,1:sh[1]/2+1] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,sh[1]/2+1:] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,1:sh[1]/2+1])))
            ph[1:sh[0]/2+1,sh[1]/2+1:] = \
                np.exp(1j*np.random.rand(sh[0]/2,sh[1]/2)*2*np.pi)
            ph[sh[0]/2+1:,1:sh[1]/2+1] = \
                np.conjugate(np.flipud(np.fliplr(ph[1:sh[0]/2+1,sh[1]/2+1:])))


##        if np.size(d) % 2 == 1:
##            ph[1:sh[0]/2] = np.random.rand(sh[0]/2-1)*2*np.pi
##            pdb.set_trace()
##            ph[sh[0]/2+1:] = np.conjugate(np.flipud(ph[1:sh[0]/2]))
##        else:
##            ph[1:(sh[0]-1)/2] = np.random.rand((sh[0]-1)/2-1)*2*np.pi
##            pdb.set_trace()
##            ph[(sh[0]+1)/2:] = np.conjugate(np.flipud(ph[1:(sh[0]-1)/2]))


##    #Fill in positive x frequencies with random phases
##    ind = freqx >= 0.
##    ph[ind] = np.exp(1j*np.random.rand(np.sum(ind))*2*np.pi)
##    #Fill in negative x frequencies with complex conjugates
##    ph[np.ceil(sh[0]/2.):,0] = np.conjugate(\
##        np.flipud(ph[:np.floor(sh[0]/2.),0]))
##    ph[0,np.ceil(sh[1]/2.):] = np.conjugate(\
##        np.flipud(ph[0,:np.floor(sh[1]/2.)]))

    return ph

def randomProfile(freq,psd):
    """
    Generate a random profile from an input PSD.
    freq should be in standard fft.fftfreq format
    psd should be symmetric as with a real signal
    sqrt(sum(psd)) will equal RMS of profile
    """
    amp = np.sqrt(psd)*len(freq)
    ph = randomizePh(amp)
    f = amp*ph
    sig = np.fft.ifft(f)
    return np.real(sig)
