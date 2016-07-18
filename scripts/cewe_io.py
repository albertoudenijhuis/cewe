#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
import tables
import os, subprocess
from copy import deepcopy
import cewe_stat


hdf5_cewe_descriptions = {
"circular"                          : "list (nvar) with True/False for each variable indicating whether circular variable of not",
"histogram_bounds"                  : "array (nvar x (nbins +1)) with histogram bounds",
"histogram_central_value"           : "array (nvar x nbins) with histogram central values",
"histogram_count"                   : "array (nvar x nbins) with histogram counts",
"nbins"                             : "number of bins, that are used for the histograms",
"nsamples"                          : "list (nvar) with number of samples for each variable",
"nvars"                             : "number of variables",
"plotname"                          : "list (nvar) with plot name for each variable",
"raw_moment_mu_x1"                  : "list (nvar) with first raw moment of samples for each variable",
"raw_moment_mu_x2"                  : "list (nvar) with second raw moment of samples for each variable",
"raw_moment_mu_x3"                  : "list (nvar) with third raw moment of samples for each variable",
"raw_moment_mu_x4"                  : "list (nvar) with fourth raw moment of samples for each variable",
"scalefactor"                       : "list (nvar) with scale factor for each variable (used for plotting)",
"scatter"                           : "a dictionary with all information of comparison of two variables A and B",
"units"                             : "list (nvar) with units for each variable",
"variables"                         : "list (nvar) with short name for each variable",
}

hdf5_cewe_descriptions_extra = {
"mean"                             : "list (nvar) with mean for each variable",
"variance"                         : "list (nvar) with variance for each variable",
"skewness"                         : "list (nvar) with skewness for each variable",
"kurtosis"                         : "list (nvar) with variance for each kurtosis",
}

hdf5_cewe_scatter_descriptions = {
"histogram2D_count"                 : "array (nvar x nvar x nbins x nbins) with 2D histogram counts",
"mixed_moment_nu"                   : "array (nvar x nvar) with mixed moment xy of samples for variable x and y",
"nsamples"                          : "array (nvar x nvar) with number of samples",
"raw_moment_mu_x1"                  : "array (nvar x nvar) with first raw moment of samples for variable x",
"raw_moment_mu_x2"                  : "array (nvar x nvar) with second raw moment of samples for variable x",
"raw_moment_mu_x3"                  : "array (nvar x nvar) with third raw moment of samples for variable x",
"raw_moment_mu_x4"                  : "array (nvar x nvar) with fourth raw moment of samples for variable x",
"raw_moment_mu_y1"                  : "array (nvar x nvar) with first raw moment of samples for variable y",
"raw_moment_mu_y2"                  : "array (nvar x nvar) with second raw moment of samples for variable y",
"raw_moment_mu_y3"                  : "array (nvar x nvar) with third raw moment of samples for variable y",
"raw_moment_mu_y4"                  : "array (nvar x nvar) with fourth raw moment of samples for variable y",
}

hdf5_cewe_scatter_descriptions_extra = {
"meanx"                             : "array (nvar x nvar) with mean of samples for variable x",
"meany"                             : "array (nvar x nvar) with mean of samples for variable y",
"variancex"                         : "array (nvar x nvar) with variance of samples for variable x",
"variancey"                         : "array (nvar x nvar) with variance of samples for variable y",
"covariancexy"                      : "array (nvar x nvar) with covariance of samples for variables x and y",
"correlationcoefficientxy"          : "array (nvar x nvar) with correlation coefficient of samples for variables x and y",
"leastsquaresfit_beta0"             : "-",
"leastsquaresfit_beta1"             : "-",
"leastsquaresfit_circ_beta0"        : "-",
"leastsquaresfit_circ_beta1"        : "-",
"leastsquaresfit_circ_beta2"        : "-",
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
    
    cewe_stat.cewe_udpate(cewe_dct)
    
    return cewe_dct


def write_cewe_hdf5_file(cewe_hdf5_filename, cewe_dct, _FillValue = -999.9):
    hdf5_temp_file = cewe_hdf5_filename[:-3] + "_temp.h5" 

    new = {}
    new['href']     = tables.openFile(hdf5_temp_file, 'w')
    new['root']     = new['href'].root
    new['scatter']  = new['href'].createGroup("/", 'scatter', 'scatter statistics')

    descr = deepcopy(hdf5_cewe_descriptions)
    #save extra data?
    if True:
        descr.update(hdf5_cewe_descriptions_extra)

    mylst = descr.keys()  
    mylst.remove('scatter')
    
    #write data
    for myvar in mylst:
        new['href'].createArray(new['root'], myvar, cewe_dct[myvar])
        new['href'].setNodeAttr(new['root']._f_getChild(myvar), 'description', descr[myvar])

    descr2 = deepcopy(hdf5_cewe_scatter_descriptions)
    if True:
        descr2.update(hdf5_cewe_scatter_descriptions_extra)
        
    cewe_scatter_dct = cewe_dct['scatter']
    for myvar in descr2.keys():
        new['href'].createArray(new['scatter'], myvar, \
            np.where(np.isnan(cewe_scatter_dct[myvar]), _FillValue, cewe_scatter_dct[myvar]) )
        new['href'].setNodeAttr(new['scatter']._f_getChild(myvar), 'description', descr2[myvar])
    
    new['href'].close()
        
    #h5 repack
    cmd = "h5repack -i '{}' -o '{}' -f GZIP=7".format(hdf5_temp_file, cewe_hdf5_filename)
    subprocess.call(cmd, shell=True)  
    os.remove(hdf5_temp_file)
