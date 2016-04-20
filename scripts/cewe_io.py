#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
import tables
import os, subprocess
from copy import deepcopy

hdf5_cewe_descriptions = {
"circular"                          : "list (nvar) with True/False for each variable indicating whether circular variable of not",
"histogram_bounds"                  : "array (nvar x (nbins +1)) with histogram bounds",
"histogram_central_value"           : "array (nvar x nbins) with histogram central values",
"histogram_count"                   : "array (nvar x nbins) with histogram counts",
"mean"                              : "list (nvar) with means of samples for each variable",
"nbins"                             : "number of bins, that are used for the histograms",
"nsamples"                          : "list (nvar) with number of samples for each variable",
"nvars"                             : "number of variables",
"plotname"                          : "list (nvar) with plot name for each variable",
"scalefactor"                       : "list (nvar) with scale factor for each variable (used for plotting)",
"scatter"                           : "a dictionary with all information of comparison of two variables A and B",
"units"                             : "list (nvar) with units for each variable",
"variables"                         : "list (nvar) with short name for each variable",
"variance"                          : "list (nvar) with variances of samples for each variable",
}

hdf5_cewe_scatter_descriptions = {
"covarianceXY"                      : "array (nvar x nvar) with covariance of variable X and Y",
"correlationcoefficientXY"          : "array (nvar x nvar) with correlation coefficient of variable X and Y",
"histogram2D_count"                 : "array (nvar x nvar x nbins x nbins) with 2D histogram counts",
"leastsquaresfit_P1"                : "array (nvar x nvar) with least squares fit parameter P1",
"leastsquaresfit_P2"                : "array (nvar x nvar) with least squares fit parameter P2",
"leastsquaresfit_P3"                : "array (nvar x nvar) with least squares fit parameter P3",
"meanX"                             : "array (nvar x nvar) with mean of variable X",
"meanY"                             : "array (nvar x nvar) with mean of variable Y",
"nsamples"                          : "array (nvar x nvar) with number of samples",
"varianceX"                         : "array (nvar x nvar) with variance of variable X",
"varianceY"                         : "array (nvar x nvar) with variance of variable Y",
}

def read_cewe_hdf5_file(cewe_hdf5_filename):
    cewe_dct = {}
    
    new = {}
    new['href']     = tables.openFile(cewe_hdf5_filename, 'r')
    new['root']     = new['href'].root
    new['scatter']  = new['root'].scatter

    mylst = deepcopy(hdf5_cewe_descriptions.keys())
    mylst.remove('nbins')
    mylst.remove('nvars')
    mylst.remove('scatter')

    for myvar in mylst:
        cewe_dct[myvar] = new['root']._f_getChild(myvar)[:]

    cewe_dct['nvars'] = cewe_dct['nsamples'].shape[0]
    cewe_dct['nbins'] = cewe_dct['histogram_central_value'].shape[1]

    cewe_scatter_dct = {}
    for myvar in hdf5_cewe_scatter_descriptions.keys():
        cewe_scatter_dct[myvar] = new['scatter']._f_getChild(myvar)[:]

    cewe_dct['scatter'] = cewe_scatter_dct
    new['href'].close()
    return cewe_dct


def write_cewe_hdf5_file(cewe_hdf5_filename, cewe_dct, _FillValue = -999.9):
    hdf5_temp_file = cewe_hdf5_filename[:-3] + "_temp.h5" 

    new = {}
    new['href']     = tables.openFile(hdf5_temp_file, 'w')
    new['root']     = new['href'].root
    new['scatter']  = new['href'].createGroup("/", 'scatter', 'scatter statistics')

    mylst = deepcopy(hdf5_cewe_descriptions.keys())
    mylst.remove('scatter')
    
    #write data
    for myvar in mylst:
        new['href'].createArray(new['root'], myvar, cewe_dct[myvar])
        new['href'].setNodeAttr(new['root']._f_getChild(myvar), 'description', hdf5_cewe_descriptions[myvar])

    cewe_scatter_dct = cewe_dct['scatter']
    for myvar in hdf5_cewe_scatter_descriptions.keys():
        new['href'].createArray(new['scatter'], myvar, \
            np.where(np.isnan(cewe_scatter_dct[myvar]), _FillValue, cewe_scatter_dct[myvar]) )

        new['href'].setNodeAttr(new['scatter']._f_getChild(myvar), 'description', hdf5_cewe_scatter_descriptions[myvar])
    
    new['href'].close()
        
    #h5 repack
    cmd = "h5repack -i '{}' -o '{}' -f GZIP=7".format(hdf5_temp_file, cewe_hdf5_filename)
    subprocess.call(cmd, shell=True)  
    os.remove(hdf5_temp_file)
