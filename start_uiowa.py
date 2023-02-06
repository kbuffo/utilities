"""
Place inside IPython startup folder to automatically import
commonly used libraries and utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.io.fits as pyfits
import pdb
import pickle
import os
import sys # set paths for package imports
util_path = r"C:\Users\kbuffo\OneDrive - University of Iowa\Research\repos\utilities"
axro_path = r"C:\Users\kbuffo\OneDrive - University of Iowa\Research\repos"
sys.path.append(util_path)
sys.path.append(axro_path)

import cylinder as cyl # utilities packages
import figure_plotting as fp
import fourier
import metrology as met
import plotting
import transformations as trf
import imaging.analysis as alsis # imaging packages
import imaging.chebFit as cheb
import imaging.fitting as fit
import imaging.man as man
import imaging.stitch as stitch
import imaging.zernikemod as zmod

import axroOptimization.cell_mapping as cellmap # axroOptimization packages
import axroOptimization.conicsolve as conic
import axroOptimization.correction_utility_functions as cuf
import axroOptimization.evaluateMirrors as eva
import axroOptimization.if_functions as iff
import axroOptimization.cell_functions as cf
# import axroOptimization.matlab_funcs as mlf
import axroOptimization.scattering as scat
import axroOptimization.scatter_v2 as scatter
import axroOptimization.solver as solver

import axroHFDFCpy.construct_connections as cc # axroHFDFCpy packages
