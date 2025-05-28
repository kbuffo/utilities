#This submodule contains image analysis routines
from matplotlib.pyplot import *
from numpy import *
import utilities.figure_plotting as fp
from matplotlib.colors import LogNorm
import pdb

def printer():
    print('Hello analysis!')

def ptov(d):
    """Return the peak to valley of an image"""
    #print(d.shape)
    #print(d)
    return nanmax(d)-nanmin(d)

def ptov_q(d_in, q=.90):
    """
    Q Peak-to-Valley: disregards the highest and lowest pixels and only consider the remaining Q Percent of pixels.
    """
    d = d_in[~np.isnan(d_in)] # exclude nans
    N_elements = int(np.round(q*d.size))
    start_idx = int((d.size - N_elements) // 2)
    d_sort = np.sort(d.flatten())
    d_trim = d_sort[start_idx:start_idx+N_elements]
    pvq = ptov(d_trim)
    return pvq

def rms(d):
    """Return the RMS of an image"""
    return sqrt(nanmean((d-nanmean(d))**2))

def funcArgIntersect(a, b):
    """
    Returns the indices where two 1 dimensional arrays intersect, even if the exact
    values are not present in both arrays.
    """
    return np.argwhere(np.diff(np.sign(a - b))).flatten()

def fitSag(d):
    """
    Compute sag to a vector by fitting a quadratic
    """
    if np.sum(np.isnan(d))==len(d):
        return np.nan
    x = np.arange(len(d))
    x = x[~np.isnan(d)]
    d = d[~np.isnan(d)]
    fit = np.polyfit(x,d,2)
    return fit[0]

def findMoments(d):
    x,y = meshgrid(arange(shape(d)[1]),arange(shape(d)[0]))
    cx = nansum(x*d)/nansum(d)
    cy = nansum(y*d)/nansum(d)
    rmsx = nansum((x-cx)**2*d)/nansum(d)
    rmsy = nansum((y-cy)**2*d)/nansum(d)
    pdb.set_trace()

    return cx,cy,sqrt(rmsx),sqrt(rmsy)

def nanflat(d):
    """
    Remove NaNs and flatten an image
    """
    d = d.flatten()
    d = d[invert(isnan(d))]
    return d

def fwhm(x,y):
    # """Compute the FWHM of an x,y vector pair"""
    # #Determine FWHM
    # maxi = np.argmax(y) #Index of maximum value
    # #Find positive bound
    # xp = x[maxi:]
    # print('max val:', y[maxi])
    # # print(np.abs(y[maxi:]-y.max()/2))
    # fwhmp = xp[np.argmin(np.abs(y[maxi:]-y.max()/2))]-x[maxi]
    # xm = x[:maxi]
    # fwhmm = x[maxi]-xm[np.argmin(np.abs(y[:maxi]-y.max()/2))]
    # return fwhmp+fwhmm

    """Compute the FWHM of an x,y vector pair"""
    #Determine FWHM
    maxi = np.nanargmax(y) #Index of maximum value
    #Find positive bound
    xp = x[maxi:]
    # print('max val:', y[maxi])
    # print(np.abs(y[maxi:]-y.max()/2))
    fwhmp = xp[np.nanargmin(np.abs(y[maxi:]-np.nanmax(y)/2))]-x[maxi]
    xm = x[:maxi]
    fwhmm = x[maxi]-xm[np.nanargmin(np.abs(y[:maxi]-np.nanmax(y)/2))]
    return fwhmp+fwhmm

class pointGetter:
    """Creates an object tied to an imshow where the user can
    accumulate a list of x,y coords by right clicking on them.
    When done, the user can press the space key.
    """
    def __init__(self, img, vmax=None, vmin=None, log=False):
        #ion()
        if not log:
            # self.fig = fp.align_figPlot(img, maxval=vmax, minval=vmin)
            self.fig = figure()
            display = imshow(img, cmap='jet', vmax=vmax, vmin=vmin)
        else:
            self.fig = figure()
        self.x = zeros(0)
        self.y = zeros(0)
        self.con = self.fig.canvas.mpl_connect('key_press_event',\
                                                self.keyEvent)
        if log:
            ax.imshow(img,norm=LogNorm())
        show()
    def keyEvent(self,event):
        if event.key==' ':
            self.x = append(self.x,event.xdata)
            self.y = append(self.y,event.ydata)
            print('Point captured!')
    def close(self):
        self.fig.canvas.mpl_disconnect(self.con)
        close(self.fig)

class peakInformation:
    def __init__(self,img):
        self.img = img
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.gx = []
        self.gy = []
        self.rmsx = []
        self.rmsy = []
        self.fig = figure()
        ax = self.fig.add_subplot(111)
        ax.imshow(img)
        self.con = self.fig.canvas.mpl_connect('button_press_event',self.clickEvent)
    def clickEvent(self,event):
        #If x0 and y0 are undefined, set them and return
        if self.x0 is None:
            #Define first point
            self.x0 = event.xdata
            self.y0 = event.ydata
            return
        #If x1 and y1 are undefined, set them and return centroid
        #Define second point
        self.x1 = event.xdata
        self.y1 = event.ydata
        #Order points properly
        x0 = min([self.x0,self.x1])
        x1 = max([self.x0,self.x1])
        y0 = min([self.y0,self.y1])
        y1 = max([self.y0,self.y1])
        #Compute centroid between coordinates and update centroid list
        cx,cy,rmsx,rmsy = findMoments(self.img[y0:y1,x0:x1])
        print('X: ' + str(cx+x0))
        print('Y: ' + str(cy+y0))
        print('RMS X: ' + str(rmsx))
        print('RMS Y: ' + str(rmsy))
        try:
            self.gx.append(cx+x0)
            self.gy.append(cy+y0)
            self.rmsx.append(rmsx)
            self.rmsy.append(rmsy)
        except:
            self.gx = [cx+x0]
            self.gy = [cy+y0]
            self.rmsx = [rmsx]
            self.rmsy = [rmsy]
        #Reset event
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
    def close(self):
        self.fig.canvas.mpl_disconnect(self.con)
        print(self.gx)
        print(self.gy)
        print(self.rmsx)
        print(self.rmsy)
        close(self.fig)

def getPoints(img, vmax=None, vmin=None, log=False):
    """This function will bring up the image, wait for the user to
    press space while hovering over points, then wait for the user to
    end by pressing enter in command line, and then return a list of x,y coords
    """
    prompt_str = """Hover mouse over fiducial marker, and press space to store its position. Exit plot window when finished."""
    print(prompt_str)
    #Create instance of pointGetter class
    p = pointGetter(img, vmax, vmin, log=log)

    #When user presses enter, close the pointGetter class and
    #return the list of coordinates
    try:
        input('Press enter...')
    except:
        pass
    x = p.x
    y = p.y
    print('# of fidicuals enetered:', len(x))
    print('')
    p.close()

    return x,y

def getSubApp(img,log=False,points=None):
    """Return a subarray defined by rectangle enclosed by two points"""
    if points is None:
        x,y = getPoints(img,log=log)
    else:
        x,y = points
    return img[y.min():y.max(),x.min():x.max()]
