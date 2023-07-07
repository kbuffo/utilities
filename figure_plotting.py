import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['savefig.facecolor']='white'
import imaging.analysis as alsis

def printer():
    print('Hello fig_plot!')

def figPlot(ims, dx, vbounds=None, colormap='Spectral', figsize=(10,10),
                title_fontsize=12, ax_fontsize=10, row_titles=(''), global_title='',
                plot_titles=(''), x_label='Azimuthal Dimension (mm)', y_label='Axial Dimension (mm)',
                cbar_title='Figure (microns)', cell_nos=None, stats=False, maxInds=None,
                share_row_cbar=False, dispRadii=False, banded_rows=False):
    """
    Returns a figure plot or set of figure plots.
    ims: 2D array or list of 2D arrays or list of lists of 2D arrays. Each list
    inside the list groups a row on the figure. Ex: figs=[[fig1, fig2], [fig3, fig4]]
    will place fig1 and fig2 on first row, and fig3 and fig4 on second row.
    dx: pixel spacing of figure image (mm/pixel)
    vbounds: list or list of lists that set the lower and upper bounds for the
    color map(s) for each plot.
    colormap: specify what color map to display the figure image with.
    figsize: set the overall figure size.
    row_titles: list of strings to specify the title for a row of plots.
    global_title: string to set the title for the overall figure.
    plot_titles: list of strings. Sets the title for each plot in the figure.
    cell_nos: int or list of ints to display the cell number for a given row.
    stats: display the peak-to-valley and rms of each plot.
    maxInds: 2D array or lists of 2D arrays. Displays a maxInd for a given figure plot.
    share_row_cbar: if true all the plots on a single row will share 1 colorbar.
    """

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.suptitle(global_title+'\n', fontsize=title_fontsize+2)
    if type(ims) is not list: ims = ((ims,),)
    N_rows = len(ims)
    N_ims = 0
    for tup in ims:
        N_ims += len(tup)
    if type(dx) is not list: dx = tuple([dx]*N_rows)
    if type(vbounds) is not list: vbounds = [None]*N_ims
    if type(row_titles) is not list: row_titles = tuple([row_titles]*N_rows)
    if type(plot_titles) is not list: plot_titles = tuple([plot_titles]*N_ims)
    if type(cbar_title) is not list: cbar_title = tuple([cbar_title]*N_rows)
    if type(cell_nos) is not list: cell_nos = tuple([cell_nos]*N_rows)
    if type(maxInds) is not list: maxInds = tuple([maxInds]*N_ims)
    subfigs = fig.subfigures(N_rows, hspace=0.05)
    im_num = 0
    for i in range(N_rows): # is index for row
        row_ims = ims[i]
        if cell_nos[i] is not None:
            cell_no = '\nCell #:'+str(cell_nos[i])
        else:
            cell_no = ''
        if type(subfigs) == np.ndarray:
            axs = format_subfigure(subfigs[i], row_titles[i], cell_no,
                                title_fontsize, banded_rows, row_ims, i)
        else:
            axs = format_subfigure(subfigs, row_titles[i], cell_no,
                                title_fontsize, banded_rows, row_ims, i)
        N_cols = len(row_ims)

        for j in range(len(row_ims)): # j is index for column inside current row
            print(len(row_ims))
            if vbounds[im_num] is None:
                vbounds[im_num] = [np.nanmin(row_ims[j]), np.nanmax(row_ims[j])]

            if len(row_ims) == 1:
                make_plot(axs, x_label, y_label, ax_fontsize, plot_titles[im_num], row_ims[j],
                'equal', dx[i], vbounds[im_num], colormap, share_row_cbar, False, cbar_title[i], j,
                N_cols, stats, maxInds[im_num], dispRadii)
            else:
                make_plot(axs[j], x_label, y_label, ax_fontsize, plot_titles[im_num], row_ims[j],
                'auto', dx[i], vbounds[im_num], colormap, share_row_cbar, True, cbar_title[i], j,
                N_cols, stats, maxInds[im_num], dispRadii)
            im_num += 1
    if not share_row_cbar:
        fig.get_layout_engine().set(hspace=0.2,
                            wspace=0.2)
    else:
            if share_row_cbar:
                fig.get_layout_engine().set(hspace=0.2,
                                    wspace=0.0)
    return fig

def format_subfigure(subfig, row_title, cell_no, title_fontsize, banded_rows, row_ims, i):
    subfig.suptitle(row_title+cell_no, fontsize=title_fontsize)
    if banded_rows and i%2==0:
        subfig.set_facecolor('0.75')
    axs = subfig.subplots(1, len(row_ims))
    return axs

def make_plot(ax, x_label, y_label, ax_fontsize, plot_title,
            data, aspect, dx, vbounds, colormap, share_row_cbar, fig_cbar,
            cbar_title, col_num, N_cols, stats, maxInd, dispRadii):
    ax.set_xlabel(x_label, fontsize=ax_fontsize)
    ax.set_ylabel(y_label, fontsize=ax_fontsize)
    ax.set_title(plot_title, fontsize=ax_fontsize+2)
    extent = mk_extent(data, dx)
    # print('vbounds received:', vbounds)
    print('vbounds:', vbounds)
    im = ax.imshow(data, extent=extent, vmin=vbounds[0], vmax=vbounds[1],
                    aspect=aspect, cmap=colormap)
    if share_row_cbar:
        if col_num+1 == N_cols:
            make_colorbar(im, ax, cbar_title, fig_cbar, ax_fontsize, ax_fontsize-2)
    else:
        make_colorbar(im, ax, cbar_title, fig_cbar, ax_fontsize, ax_fontsize-2)
    if stats:
        rms = alsis.rms(data)
        ptov = alsis.ptov(data)
        ylim = ax.get_ylim()[1]
        xlim = ax.get_xlim()[1]
        disp_txt = "RMS: {:.2f} {}\nPV: {:.1f} {}".format(rms, 'um', ptov, 'um')
        ax.text(0.05, 0.13, disp_txt, fontsize=ax_fontsize, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.65))
    if maxInd is not None:
        pass
    if dispRadii:
        large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=ax_fontsize, color='red',
                                ha='center', va='center', transform=ax.transAxes)
        small_R_text = ax.text(0.5, 0.9255, 'Smaller R', fontsize=ax_fontsize, color='red',
                                ha='center', va='center', transform=ax.transAxes)


def make_colorbar(im, ax, cbar_title, fig_cbar, cbar_fontsize, tick_fntsz):
    """
    Takes in an imshow image and associated axes and adds a color bar to it.
    """
    divider = make_axes_locatable(ax)
    if fig_cbar:
        cbar = plt.colorbar(im, ax=ax, pad=0.05)
    else:
        cax = divider.append_axes("right", size="7%", pad="5%")
        cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_title, fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=tick_fntsz)

def mk_colormesh(ax, d, minval, maxval, colormap):
    """
    Generate colormesh plot given current axis, data, colormap,
    and min and max values for colorbar range.
    """
    # print('THE MINVAL IS {}'.format(minval))
    height_map = ax.pcolormesh(d, cmap=colormap, vmax=maxval, vmin=minval)
    # if maxval and minval:
    #     print('got minval and maxval')
    #     height_map = ax.pcolormesh(d, cmap=colormap, vmax=maxval, vmin=minval)
    # elif maxval and not minval:
    #     height_map = ax.pcolormesh(d, cmap=colormap, vmax=maxval)
    # elif minval and not maxval:
    #     height_map = ax.pcolormesh(d, cmap=colormap, vmax=minval)
    # else:
    #     height_map = ax.pcolormesh(d, cmap=colormap)
    return height_map

def get_tickvals(d, dx, xtick_vals, ytick_vals):
    """
    Get tick locations for x and y axis.
    xtick_vals and ytick_vals are the values of the desired
    ticks in mm, and the function converts from pixels to mm
    using d and dx to return where those tick values are located.
    """
    # find center of both axis
    y_shape = d.shape[0]
    x_shape = d.shape[1]
    y0 = y_shape/2
    x0 = x_shape/2
    # calculate tick mark locations
    # convert from pixels to mm
    xtick_loc = []
    ytick_loc = []
    for i in xtick_vals:
        xtick_loc.append(x0 + (i/dx))
    for i in ytick_vals:
        ytick_loc.append(y0 + (i/dx))

    return (xtick_loc, ytick_loc)

def old_figPlot(d_in, dx, xtick_vals=None, ytick_vals=None, plotsize=(7,6),
                title=None, title_fntsz=12, x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)', cbar_title='Figure (microns)',
                ax_fntsz=10, tick_fntsz=8, stats=False, units='um',
                colormap='Spectral', ax=None,
                vmin=None, vmax=None, returnax=False, com=None):
    """
    Generate figure plot using loaded 4D interferometric data.
    Takes in 4D array data "d" and pixel density "dx".
    Specify tick mark locations in mm, plotsize, titles,
    fontsizes, and peak-to-valley and rms values.
    """
    d = np.copy(d_in)
    if units == 'nm':
        d *= 1000.
    if ax == None:
        fig, ax = plt.subplots(figsize=plotsize)
    extent = mk_extent(d, dx)
    im = ax.imshow(d, extent=extent, vmin=vmin, vmax=vmax,
                    aspect='auto', cmap=colormap)
    if xtick_vals:
        ax.set_xticks(xtick_vals)
    if ytick_vals:
        ax.set_yticks(ytick_vals)
    ax.set_xlabel(x_title, fontsize=ax_fntsz)
    ax.set_ylabel(y_title, fontsize=ax_fntsz)
    if title:
        ax.set_title(title, fontsize=title_fntsz)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz)
    cbar.ax.tick_params(labelsize=tick_fntsz)

    if stats:
        rms = alsis.rms(d)
        ptov = alsis.ptov(d)
        ylim = ax.get_ylim()[1]
        xlim = ax.get_xlim()[1]
        disp_txt = "RMS: {:.2f} {}\nPV: {:.1f} {}".format(rms, units, ptov, units)
        plt.text(0.05, 0.13, disp_txt, fontsize=ax_fntsz, transform=ax.transAxes)
    if com:
        ycom, xcom = -(com[0]-100), -com[1]
        print('got com:', com)
        ax.axhline(ycom, lw=6, color='black')
        ax.axvline(xcom, lw=6, color='black')
    if ax == None:
        if returnax:
            return ax
        else:
            return fig

def figPlot2(cyl_data, cyl_dx, title, colormap='Spectral'):
    """
    Generate figure plot using Casey's method.
    cyl_data is loaded 4D interferometric data and
    pixel density cyl_data.
    Specify title to display on figure.
    """
    fig = plt.figure(figsize = (9,9))
    extent = [-np.shape(cyl_data)[1]*cyl_dx/2,np.shape(cyl_data)[1]*cyl_dx/2,-np.shape(cyl_data)[0]*cyl_dx/2,np.shape(cyl_data)[0]*cyl_dx/2]
    fs = 12

    ax = plt.gca()
    im = ax.imshow(cyl_data,extent = extent,aspect = 'auto',cmap = colormap)
    ax.set_xlabel('Azimuthal Dimension (mm)',fontsize = fs)
    ax.set_ylabel('Axial Dimension (mm)',fontsize = fs)
    ax.set_title(title,fontsize = fs*1.25)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im,cax = cax)
    cbar.set_label('Figure (microns)',fontsize = fs)

    img = cyl_data
    plt.text(0.05,0.13,'RMS: ' + "{:.3}".format(np.nanstd(img)) + ' um',horizontalalignment = 'left',verticalalignment = 'center',transform = ax.transAxes,fontsize = fs)
    plt.text(0.05,0.09,'PV: ' + "{:.3}".format(np.nanmax(img) -np.nanmin(img)) + ' um',horizontalalignment = 'left',verticalalignment = 'center',transform = ax.transAxes, fontsize = fs)

    return fig

def align_figPlot(d, dx=None, xtick_vals=None, ytick_vals=None, plotsize=(7,6),
                title=None, title_fntsz=14, x_title=None,
                y_title=None, cbar_title=None, ax_fntsz=12,
                tick_fntsz=10, stats=False, maxval=None, minval=None,
                colormap='jet'):
    """
    Generate figure plot using legendre subtracted alignment data.
    Takes in 4D array data "d" and pixel density "dx".
    Specify tick mark locations in mm, plotsize, titles,
    fontsizes, and peak-to-valley and rms values.
    maxval and minval specifies the range of the colorbar.
    """
    fig, ax = plt.subplots(figsize=plotsize)
    d = np.flipud(d)
    height_map = mk_colormesh(ax, d, minval, maxval, colormap)
    if dx and xtick_vals:
        if ytick_vals == None: ytick_vals = xtick_vals
        xtick_loc, ytick_loc = get_tickvals(d, dx, xtick_vals, ytick_vals)
        ax.set_xticks(xtick_loc)
        ax.set_yticks(ytick_loc)
        ax.set_xticklabels(xtick_vals, fontsize=tick_fntsz)
        ax.set_yticklabels(ytick_vals, fontsize=tick_fntsz)
    ax.set_xlabel(x_title, fontsize=ax_fntsz)
    ax.set_ylabel(y_title, fontsize=ax_fntsz)
    if title:
        ax.set_title(title, fontsize=title_fntsz)
    cbar = fig.colorbar(height_map)
    cbar.set_label(cbar_title, fontsize=ax_fntsz)
    cbar.ax.tick_params(labelsize=tick_fntsz)
    if stats:
        rms = alsis.rms(d)
        ptov = alsis.ptov(d)
        ylim = ax.get_ylim()[1]
        xlim = ax.get_xlim()[1]
        disp_txt = "RMS: {:.2f} um\nPV: {:.1f} um".format(rms, ptov)
        plt.text(0.05*xlim, 0.1*ylim, disp_txt, fontsize=ax_fntsz)

    return fig

def figPlot_grid(d_ls, dx_ls, xtick_vals=None, ytick_vals=None, gridsize=(1,1),
                figsize=(14,14), title_ls=None,
                global_title=None, title_fntsz=12, x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)', cbar_title='Figure (microns)',
                ax_fntsz=10, tick_fntsz=8, stats=False, units='um',
                colormap='Spectral',
                vmin=None, vmax=None, returnax=False, com=None):
    """
    Generate a grid of figure plots using loaded 4D interferometric data.
    Takes in 4D array data "d" and pixel density "dx".
    Specify tick mark locations in mm, plotsize, titles,
    fontsizes, and peak-to-valley and rms values.
    """
    nrows, ncols = gridsize[0], gridsize[1]
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    # fig = plt.figure(figsize=figsize)
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i == len(d_ls):
                break
            else:
                d = d_ls[i]
                dx = dx_ls[i]
                title = title_ls[i]
                if nrows == 1:
                    ax = axs[c]
                elif ncols == 1:
                    ax = axs[r]
                else:
                    ax = axs[r, c]
                figPlot(d, dx, xtick_vals=xtick_vals, ytick_vals=ytick_vals, plotsize=figsize,
                                title=title, title_fntsz=title_fntsz, x_title=x_title,
                                y_title=y_title, cbar_title=cbar_title, ax=ax,
                                ax_fntsz=ax_fntsz, tick_fntsz=ax_fntsz, stats=stats, units=units,
                                colormap=colormap, vmin=vmin, vmax=vmax, returnax=returnax, com=com)
                i += 1
    fig.suptitle(global_title, fontsize=title_fntsz)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace = 0.4, wspace = 0.5)
    return fig

def save_figLocal(fig, dir, filename):
    """
    Saves figure to dir\Images as PNG. If directory doesn't exist,
    creates it and saves figure. Input os.getcwd() for dir to check current
    working directory.
    """
    save_path = os.path.join(dir, 'Images')
    if os.path.exists(save_path):
        fig.savefig(os.path.join(save_path, filename+'.png'), format='png')
    else:
        os.mkdir(save_path)
        fig.savefig(os.path.join(save_path, filename+'.png'), format='png')

def mk_extent(d, dx):
    img_shp = np.shape(d)
    return [-img_shp[1]*dx/2, img_shp[1]*dx/2, -img_shp[0]*dx/2, img_shp[0]*dx/2]
