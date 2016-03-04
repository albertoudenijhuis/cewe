#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
import tables
import os, subprocess

hdf5_cewe_descriptions = {
"angular"							: "list (nvar) with True/False for each variable indicating whether angular variable of not",
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
"covarianceAB"                      : "array (nvar x nvar) with covariance of variable A and B",
"correlationcoefficientAB"          : "array (nvar x nvar) with correlation coefficient of variable A and B",
"histogram2D_count"                 : "array (nvar x nvar x nbins x nbins) with 2D histogram counts",
"leastsquaresfit_C"                 : "array (nvar x nvar) with least squares fit parameter C",
"leastsquaresfit_D"                 : "array (nvar x nvar) with least squares fit parameter D",
"meanA"                             : "array (nvar x nvar) with mean of variable A",
"meanB"                             : "array (nvar x nvar) with mean of variable B",
"nsamples"                          : "array (nvar x nvar) with number of samples",
"varianceA"                         : "array (nvar x nvar) with variance of variable A",
"varianceB"                         : "array (nvar x nvar) with variance of variable B",
}

def read_cewe_hdf5_file(cewe_hdf5_filename):
    cewe_dct = {}
    
    new = {}
    new['href'] 	= tables.openFile(cewe_hdf5_filename, 'r')
    new['root'] 	= new['href'].root
    new['scatter'] 	= new['root'].scatter

    for myvar in [
        'angular',
        'histogram_bounds',
        'histogram_central_value',
        'histogram_count',
        'mean',
        'nsamples',
        'plotname',
        'scalefactor',
        'units',
        'variables',
        'variance',
        ]:
        cewe_dct[myvar] = new['root']._f_getChild(myvar)[:]

    cewe_dct['nvars'] = cewe_dct['nsamples'].shape[0]
    cewe_dct['nbins'] = cewe_dct['histogram_central_value'].shape[1]

    cewe_scatter_dct = {}
    for myvar in [
        'covarianceAB',
        'correlationcoefficientAB',
        'histogram2D_count',
        'leastsquaresfit_C',
        'leastsquaresfit_D',
        'meanA',
        'meanB',
        'nsamples',
        'varianceA',
        'varianceB',
        ]:
        cewe_scatter_dct[myvar] = new['scatter']._f_getChild(myvar)[:]

    cewe_dct['scatter'] = cewe_scatter_dct
    new['href'].close()
    return cewe_dct


def write_cewe_hdf5_file(cewe_hdf5_filename, cewe_dct, _FillValue = -999.9):
    hdf5_temp_file = cewe_hdf5_filename[:-3] + "_temp.h5" 

    new = {}
    new['href'] 	= tables.openFile(hdf5_temp_file, 'w')
    new['root'] 	= new['href'].root
    new['scatter'] 	= new['href'].createGroup("/", 'scatter', 'scatter statistics')

    #write data
    for myvar in [
        'angular',
        'histogram_bounds',
        'histogram_central_value',
        'histogram_count',
        'mean',
        'nbins',
        'nsamples',
        'nvars',
        'plotname',
        'scalefactor',
        'units',
        'variables',
        'variance',
        ]:
        new['href'].createArray(new['root'], myvar, cewe_dct[myvar])
        new['href'].setNodeAttr(new['root']._f_getChild(myvar), 'description', hdf5_cewe_descriptions[myvar])

    cewe_scatter_dct = cewe_dct['scatter']
    for myvar in [
        'covarianceAB',
        'correlationcoefficientAB',
        'histogram2D_count',
        'leastsquaresfit_C',
        'leastsquaresfit_D',
        'meanA',
        'meanB',
        'nsamples',
        'varianceA',
        'varianceB',
        ]:    
        new['href'].createArray(new['scatter'], myvar, \
            np.where(np.isnan(cewe_scatter_dct[myvar]), _FillValue, cewe_scatter_dct[myvar]) )

        new['href'].setNodeAttr(new['scatter']._f_getChild(myvar), 'description', hdf5_cewe_scatter_descriptions[myvar])
    
    new['href'].close()
        
    #h5 repack
    cmd = "h5repack -i '{}' -o '{}' -f GZIP=7".format(hdf5_temp_file, cewe_hdf5_filename)
    subprocess.call(cmd, shell=True)  
    os.remove(hdf5_temp_file)
