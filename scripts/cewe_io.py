#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
import tables
import os, subprocess

def read_stats(h5statsfile):
    dct         = {}
    
    new = {}
    new['href'] = tables.openFile(h5statsfile, 'r')
    new['root'] = new['href'].root
    new['scatter'] =  new['root'].scatter

    for myvar in [
        'variables',
        'plotname',
        'units',
        'scalefactor',
        'angular',
        'n',
        'mean',
        'variance',
        'distribution_norm_ppt',
        'distribution_val',
        'distribution_cdf',
        ]:
        dct[myvar] = new['root']._f_getChild(myvar)[:]

    dct_scatter = {}
    for myvar in [
        'n',
        'meanA',
        'meanB',
        'varianceA',
        'varianceB',
        'covarianceAB',
        'correlationcoefficientAB',
        'leastsquaresfit_A',
        'leastsquaresfit_B',
        'histogram2D_bounds',
        'histogram2D_central_value',
        'histogram2D_values',
        ]:
        dct_scatter[myvar] = new['scatter']._f_getChild(myvar)[:]

    dct['scatter'] = dct_scatter
    new['href'].close()
    return dct


def write_stats(h5statsfile, dct, _FillValue = -999.9):
    h5tmpfile = h5statsfile[:-3] + "_temp.h5" 

    new = {}
    new['href'] = tables.openFile(h5tmpfile, 'w')
    new['root'] = new['href'].root
    new['scatter'] =  new['href'].createGroup("/", 'scatter', 'scatter statistics')

    #write data
    for myvar in [
        'variables',
        'plotname',
        'units',
        'scalefactor',
        'angular',
        'n',
        'mean',
        'variance',
        'distribution_norm_ppt',
        'distribution_val',
        'distribution_cdf',
        ]:
        new['href'].createArray(new['root'], myvar, dct[myvar])

    #write data
    scatterdct = dct['scatter']
    for myvar in [
        'n',
        'meanA',
        'meanB',
        'varianceA',
        'varianceB',
        'covarianceAB',
        'correlationcoefficientAB',
        'leastsquaresfit_A',
        'leastsquaresfit_B',
        'histogram2D_bounds',
        'histogram2D_central_value',
        'histogram2D_values',
        ]:    
        new['href'].createArray(new['scatter'], myvar, \
            np.where(np.isnan(scatterdct[myvar]), _FillValue, scatterdct[myvar]) )
    
    new['href'].close()
        
    #h5 repack
    cmd = "h5repack -i '{}' -o '{}' -f GZIP=7".format(h5tmpfile, h5statsfile)
    subprocess.call(cmd, shell=True)  
    os.remove(h5tmpfile)
