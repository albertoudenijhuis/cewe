#!/usr/bin/env python

__readme = \
"""
Part of the CEWE library. See cewe.py for the general details.

For calculations CEWE dictionaries are used.

A CEWE dictionary should always contain the following:
"circular"                          : "list (nvar) with True/False for each variable indicating whether circular variable of not",
"histogram_bounds"                  : "array (nvar x (nbins +1)) with histogram bounds",
"histogram_central_value"           : "array (nvar x nbins) with histogram central values",
"histogram_count"                   : "array (nvar x nbins) with histogram counts",
"nbins"                             : "number of bins, that are used for the histograms",
"nsamples"                          : "list (nvar) with number of samples for each variable",
"nvars"                             : "number of variables",
"plotname"                          : "list (nvar) with plot name for each variable",
"raw_moment_mu_x1"                   : "list (nvar) with first raw moment of samples for each variable",
"raw_moment_mu_x2"                   : "list (nvar) with second raw moment of samples for each variable",
"raw_moment_mu_x3"                   : "list (nvar) with third raw moment of samples for each variable",
"raw_moment_mu_x4"                   : "list (nvar) with fourth raw moment of samples for each variable",
"scalefactor"                       : "list (nvar) with scale factor for each variable (used for plotting)",
"scatter"                           : "a dictionary with all information of comparison of two variables A and B",
"units"                             : "list (nvar) with units for each variable",
"variables"                         : "list (nvar) with short name for each variable",

A CEWE scatter dictionary should always contain the following:
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

To create a CEWE ditrionary, a dataset dictionary is used as input.

A dataset ditrionary contains
- A                                 : Variables that are a dictionary
- B                                 : ...
- C                                 : ...

A variable dictionary contains (e.g. 'A')
- circular                           : True/False indicating whether circular variable or not
- llim                              : lower limit for creation of bins
- plotname                          : name used for plotting
- samples                           : samples of the data
- scalefactor                       : scale factor used for plotting
- ulim                              : uper limit for creation of bins
- units                             : units used for plotting

Most important Functions:
create_cewe_dct                     : create CEWE dictionary given a dataset dictionary
cewe_udpate							: calculate related statistics.
combine_cewe_dct                    : function that combines two CEWE dictionaries.
"""


import warnings; warnings.simplefilter("ignore")
import numpy as np
from scipy import stats
from copy import deepcopy
from scipy.interpolate import interp1d

#default options for the creation of a CEWE dictionary. Can be modified by the user.
cewe_default_opts = {'nbins':50, '_FillValue': [-999.9]}    

def create_cewe_dct(input_dataset, input_cewe_opts={}):
    #options
    cewe_opts = deepcopy(cewe_default_opts)
    cewe_opts.update(input_cewe_opts)
    _FillValue      = cewe_opts['_FillValue'][0]

    #copy dataset
    dataset = deepcopy(input_dataset)

    #add additional variables for circular variables
    lst_vars = dataset.keys()
    for myvar in lst_vars:
        if dataset[myvar]['circular']:
            fct = (2. * np.pi) / (dataset[myvar]['ulim'] - dataset[myvar]['llim'])
            dataset[myvar+'_cartesian_cos'] = {
                'circular': False,
                'llim': -1.,
                'ulim': 1.,
                'plotname': 'cosinus of ' + dataset[myvar]['plotname'],
                'units': '-',
                'scalefactor': 1.,
                'samples': np.where(
                                dataset[myvar]['samples'] == _FillValue,
                                _FillValue,
                                np.cos(fct * dataset[myvar]['samples'])),
                }
            dataset[myvar+'_cartesian_sin'] = {
                'circular': False,
                'llim': -1.,
                'ulim': 1.,
                'plotname': 'sinus of ' + dataset[myvar]['plotname'],
                'units': '-',
                'scalefactor': 1.,
                'samples': np.where(
                                dataset[myvar]['samples'] == _FillValue,
                                _FillValue,
                                np.sin(fct * dataset[myvar]['samples'])),
                }
    
    #TBD: check dataset dictionary
    #...
    

    #Fill Values
    _FillValues     = cewe_opts['_FillValue']
    _FillValue      = cewe_opts['_FillValue'][0]

    #generate dictionary
    cewe_dct = {}

    #take over list of variables    
    cewe_dct['variables']       = sorted(dataset.keys())
    lst_vars                    = np.array(cewe_dct['variables'])

    #set numbers
    cewe_dct['nvars']           = len(cewe_dct['variables'])
    cewe_dct['nbins']           = cewe_opts['nbins']

    #create dummy lists
    cewe_dct['plotname']        = deepcopy(cewe_dct['variables'])
    cewe_dct['scalefactor']     = deepcopy(cewe_dct['variables'])
    cewe_dct['units']           = deepcopy(cewe_dct['variables'])
    
    #create dummy lists of integers
    cewe_dct['circular']        = np.zeros(cewe_dct['nvars'], dtype=np.int) + _FillValue
    cewe_dct['nsamples']        = np.zeros(cewe_dct['nvars'], dtype=np.int) + _FillValue
    
    #create dummy list of floats
    cewe_dct['histogram_bounds']        = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'] + 1)) + _FillValue
    cewe_dct['histogram_central_value'] = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'])) + _FillValue
    cewe_dct['histogram_count']         = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'])) + _FillValue
    cewe_dct['raw_moment_mu_x1']         = np.zeros(cewe_dct['nvars']) + _FillValue
    cewe_dct['raw_moment_mu_x2']         = np.zeros(cewe_dct['nvars']) + _FillValue
    cewe_dct['raw_moment_mu_x3']         = np.zeros(cewe_dct['nvars']) + _FillValue
    cewe_dct['raw_moment_mu_x4']         = np.zeros(cewe_dct['nvars']) + _FillValue

    #statistics, pass for linear statistics
    for mynr in range(cewe_dct['nvars']): 
        myvar                                       = cewe_dct['variables'][mynr]
        mycirc                                      = dataset[myvar]['circular']

        #set metadata
        cewe_dct['circular'][mynr]                  = 1 if (mycirc) else 0
        cewe_dct['histogram_bounds'][mynr]          = np.linspace(dataset[myvar]['llim'], dataset[myvar]['ulim'], cewe_dct['nbins'] + 1)  
        cewe_dct['histogram_central_value'][mynr]   = (cewe_dct['histogram_bounds'][mynr][1:] + cewe_dct['histogram_bounds'][mynr][:-1]) / 2.
        cewe_dct['plotname'][mynr]                  = dataset[myvar]['plotname']
        cewe_dct['scalefactor'][mynr]               = 1. if (dataset[myvar]['scalefactor'] <= 0) else (1. * dataset[myvar]['scalefactor'])
        cewe_dct['units'][mynr]                     = dataset[myvar]['units']

        #okdata
        myok                        = ok(dataset[myvar]['samples'], cewe_opts)
        cewe_dct['nsamples'][mynr]  = np.sum(myok)

        mysamples = np.compress(myok, dataset[myvar]['samples'])                             
            
        if not mycirc:
            #calculate raw moments
            for i in [1,2,3,4]:
                cewe_dct['raw_moment_mu_x'+str(i)][mynr] = raw_moment(mysamples, i)
            
        #histogram
        myhistogram, dummy                  = np.histogram(mysamples, bins=cewe_dct['histogram_bounds'][mynr])
        cewe_dct['histogram_count'][mynr]   = deepcopy(myhistogram)

        del mysamples

                
            
    #scatter statistics.
    cewe_scatter_dct = {}
    
    sh1 = (cewe_dct['nvars'], cewe_dct['nvars'])
    sh2 = (cewe_dct['nvars'], cewe_dct['nvars'], cewe_dct['nbins'], cewe_dct['nbins'])

    cewe_scatter_dct['histogram2D_count']           = np.zeros(sh2) + _FillValue
    cewe_scatter_dct['mixed_moment_nu']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['nsamples']                    = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_x1']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_x2']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_x3']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_x4']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_y1']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_y2']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_y3']             = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['raw_moment_mu_y4']             = np.zeros(sh1) + _FillValue
    
    #scatter statistics, pass for linear statistics
    for mynrx in range(cewe_dct['nvars']):   
        myvarx = cewe_dct['variables'][mynrx]
        for mynry in range(cewe_dct['nvars']):  
            myvary = cewe_dct['variables'][mynry]
        
            myokxy =  ok(dataset[myvarx]['samples'], cewe_opts) & ok(dataset[myvary]['samples'], cewe_opts)
            mycircx  = cewe_dct['circular'][mynrx]
            mycircy  = cewe_dct['circular'][mynry]
          
            cewe_scatter_dct['nsamples'][mynrx, mynry] = np.sum(myokxy)

            mydatax = np.compress(myokxy, dataset[myvarx]['samples'])                              
            mydatay = np.compress(myokxy, dataset[myvary]['samples'])                              
            
            #calculate raw moments
            if not mycircx:
                for i in [1,2,3,4]:
                    cewe_scatter_dct['raw_moment_mu_x'+str(i)][mynrx, mynry] = raw_moment(mydatax, i)
                                    
            if not mycircy:
                for i in [1,2,3,4]:
                    cewe_scatter_dct['raw_moment_mu_y'+str(i)][mynrx, mynry] = raw_moment(mydatay, i)

            #calculate mixed moments
            if ((not mycircx) and (not mycircy)):
                cewe_scatter_dct['mixed_moment_nu'][mynrx, mynry]  = mixed_moment(mydatax, mydatay)

            #2D histogram.
            x_llim, x_ulim = dataset[myvarx]['llim'], dataset[myvarx]['ulim']
            y_llim, y_ulim = dataset[myvary]['llim'], dataset[myvary]['ulim']

            bins = [ np.linspace(x_llim, x_ulim, cewe_dct['nbins'] + 1), np.linspace(y_llim, y_ulim, cewe_dct['nbins'] + 1) ]

            dummy1, dummy2, dummy3 = np.histogram2d(mydatax, mydatay , bins, normed=False )
            cewe_scatter_dct['histogram2D_count'][mynrx, mynry] = np.transpose(dummy1) 

    #set scatter dictionary
    cewe_dct['scatter'] = cewe_scatter_dct

    #update
    cewe_udpate(cewe_dct)
       
    return cewe_dct             

def cewe_udpate(cewe_dct, input_cewe_opts={}):
    #options
    cewe_opts = deepcopy(cewe_default_opts)
    cewe_opts.update(input_cewe_opts)
    _FillValue      = cewe_opts['_FillValue'][0]
    
    cewe_scatter_dct    = cewe_dct['scatter']
    lst_vars            = np.array(cewe_dct['variables'])

    #
    #
    #statistics, pass for linear
    nvars = cewe_dct['nvars']

    #
    nsamples = cewe_dct['nsamples']
    mu_x1 = cewe_dct['raw_moment_mu_x1']
    mu_x2 = cewe_dct['raw_moment_mu_x2']
    mu_x3 = cewe_dct['raw_moment_mu_x3']
    mu_x4 = cewe_dct['raw_moment_mu_x4']

    myfilter =    ((mu_x1 == _FillValue) | (nsamples < 2))

    cewe_dct['mean']        = deepcopy(mu_x1)
    
    cewe_dct['variance']    = \
        np.where(myfilter, _FillValue,
            (nsamples / (nsamples - 1.)) * (mu_x2 - (mu_x1**2.))
            )
            
    cewe_dct['skewness']    = \
        np.where(myfilter, _FillValue,
            (mu_x3 - (3. * mu_x2 * mu_x1) + (2. * (mu_x1 ** 3.)))
            /
            ((mu_x2 - (mu_x1 **2.)) ** (3./2.))
            )

    cewe_dct['kurtosis']    = \
        np.where(myfilter, _FillValue,
            (mu_x4 - (4. * mu_x1 * mu_x3) + (6. * (mu_x1 ** 2.) * mu_x2) - (3. * (mu_x1**4.)))
            /
            ((mu_x2 - (mu_x1 **2.)) ** 2.)
            )
    
    #statistics, pass for circular
    for mynr in range(nvars): 
        myvar                                       = cewe_dct['variables'][mynr]
        mycirc                                      = cewe_dct['circular'][mynr]

        if (cewe_dct['nsamples'][mynr] > 1):
            if mycirc:
                mynr_cos                            = np.argmax(lst_vars == (myvar+'_cartesian_cos'))
                mynr_sin                            = np.argmax(lst_vars == (myvar+'_cartesian_sin'))
                
                #calculate mean and variance
                cewe_dct['mean'][mynr]              = circ_mean(cewe_dct['mean'][mynr_cos], cewe_dct['mean'][mynr_sin])
                cewe_dct['variance'][mynr]          = circ_variance(cewe_dct['mean'][mynr_cos], cewe_dct['mean'][mynr_sin])

    #scatter statistics, pass for linear
    cewe_scatter_dct['meanx'] = deepcopy(cewe_scatter_dct['raw_moment_mu_x1'])
    cewe_scatter_dct['meany'] = deepcopy(cewe_scatter_dct['raw_moment_mu_y1'])

    filter_samples = (cewe_scatter_dct['nsamples'] < 2) 
    myfilterx = filter_samples |  (cewe_scatter_dct['raw_moment_mu_x1'] == _FillValue)
    myfiltery = filter_samples |  (cewe_scatter_dct['raw_moment_mu_y1'] == _FillValue)
    myfilterxy = filter_samples |  (cewe_scatter_dct['mixed_moment_nu'] == _FillValue)

    cewe_scatter_dct['variancex']    = \
        np.where(myfilterx, _FillValue,
            ((cewe_scatter_dct['nsamples']) / (cewe_scatter_dct['nsamples'] - 1.)) * (cewe_scatter_dct['raw_moment_mu_x2'] - (cewe_scatter_dct['raw_moment_mu_x1']**2.))
            )
    cewe_scatter_dct['variancey']    = \
        np.where(myfiltery, _FillValue,
            ((cewe_scatter_dct['nsamples']) / (cewe_scatter_dct['nsamples'] - 1.)) * (cewe_scatter_dct['raw_moment_mu_y2'] - (cewe_scatter_dct['raw_moment_mu_y1']**2.))
            )
    cewe_scatter_dct['covariancexy']    = \
        np.where(myfilterxy, _FillValue,
            ((cewe_scatter_dct['nsamples']) / (cewe_scatter_dct['nsamples'] - 1.)) * (cewe_scatter_dct['mixed_moment_nu'] - (cewe_scatter_dct['raw_moment_mu_x1'] * cewe_scatter_dct['raw_moment_mu_y1']))
            )
                
    #scatter statistics, first pass for circular
    for mynrx in range(nvars):   
        myvarx = cewe_dct['variables'][mynrx]
        for mynry in range(nvars):  
            myvary = cewe_dct['variables'][mynry]
        
            mycircx  = cewe_dct['circular'][mynrx]
            mycircy  = cewe_dct['circular'][mynry]

            if mycircx:
                mynrx_cos                           = np.argmax(lst_vars == (myvarx+'_cartesian_cos'))
                mynrx_sin                           = np.argmax(lst_vars == (myvarx+'_cartesian_sin'))
            if mycircy:
                mynry_cos                           = np.argmax(lst_vars == (myvary+'_cartesian_cos'))
                mynry_sin                           = np.argmax(lst_vars == (myvary+'_cartesian_sin'))

            #mean
            if mycircx:
                cewe_scatter_dct['meanx'][mynrx, mynry] = circ_mean(cewe_scatter_dct['meanx'][mynrx_cos, mynry], cewe_scatter_dct['meanx'][mynrx_sin, mynry])
            if mycircy:
                cewe_scatter_dct['meany'][mynrx, mynry] = circ_mean(cewe_scatter_dct['meany'][mynrx, mynry_cos], cewe_scatter_dct['meany'][mynrx, mynry_sin])

    #scatter statistics, second pass for circular
    for mynrx in range(nvars):   
        myvarx = cewe_dct['variables'][mynrx]
        for mynry in range(nvars):  
            myvary = cewe_dct['variables'][mynry]
        
            mycircx  = cewe_dct['circular'][mynrx]
            mycircy  = cewe_dct['circular'][mynry]

            if mycircx:
                mynrx_cos                           = np.argmax(lst_vars == (myvarx+'_cartesian_cos'))
                mynrx_sin                           = np.argmax(lst_vars == (myvarx+'_cartesian_sin'))
            if mycircy:
                mynry_cos                           = np.argmax(lst_vars == (myvary+'_cartesian_cos'))
                mynry_sin                           = np.argmax(lst_vars == (myvary+'_cartesian_sin'))
                
            #variance
            if mycircx:
                cewe_scatter_dct['variancex'][mynrx, mynry] = circ_variance(cewe_scatter_dct['meanx'][mynrx_cos, mynry], cewe_scatter_dct['meanx'][mynrx_sin, mynry])
            if mycircy:
                cewe_scatter_dct['variancey'][mynrx, mynry] = circ_variance(cewe_scatter_dct['meany'][mynrx, mynry_cos], cewe_scatter_dct['meany'][mynrx, mynry_sin])

            #covariance
            if mycircx:
                cewe_scatter_dct['covariancexy'][mynrx, mynry] = _FillValue
            if mycircy:
                cewe_scatter_dct['covariancexy'][mynrx, mynry] = _FillValue


    #~
    #~
    #correlation coefficients
    #least squares fit parameters
    cewe_scatter_dct['correlationcoefficientxy']        = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue
    cewe_scatter_dct['leastsquaresfit_beta0']           = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue
    cewe_scatter_dct['leastsquaresfit_beta1']           = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue

    cewe_scatter_dct['leastsquaresfit_circ_beta0']      = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue
    cewe_scatter_dct['leastsquaresfit_circ_beta1']      = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue
    cewe_scatter_dct['leastsquaresfit_circ_beta2']      = np.zeros((cewe_dct['nvars'], cewe_dct['nvars'])) + _FillValue

    #pass for linear
    for mynrx in range(nvars):                
        myvarx = cewe_dct['variables'][mynrx]
        for mynry in range(nvars): 
            myvary = cewe_dct['variables'][mynry]
        
            mycircx  = cewe_dct['circular'][mynrx]
            mycircy  = cewe_dct['circular'][mynry]

            #correlation coefficient
            if not (mycircx or mycircy):
                cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry] = cewe_scatter_dct['covariancexy'][mynrx, mynry] / np.sqrt(cewe_scatter_dct['variancex'][mynrx, mynry] * cewe_scatter_dct['variancey'][mynrx, mynry])

            #least squares fit parameters
            if not (mycircx or mycircy):
                cewe_scatter_dct['leastsquaresfit_beta0'][mynrx, mynry], cewe_scatter_dct['leastsquaresfit_beta1'][mynrx, mynry] = \
                    lsf_parameters(
                        cewe_scatter_dct['meanx'][mynrx, mynry],
                        cewe_scatter_dct['meany'][mynrx, mynry],
                        cewe_scatter_dct['variancex'][mynrx, mynry],
                        cewe_scatter_dct['variancey'][mynrx, mynry],
                        cewe_scatter_dct['covariancexy'][mynrx, mynry],
                    )                   
    
    #pass for circular
    for mynrx in range(nvars):                
        myvarx = cewe_dct['variables'][mynrx]
        for mynry in range(nvars): 
            myvary = cewe_dct['variables'][mynry]
        
            mycircx  = cewe_dct['circular'][mynrx]
            mycircy  = cewe_dct['circular'][mynry]
            
            if mycircx:
                mynrx_cos                           = np.argmax(lst_vars == (myvarx+'_cartesian_cos'))
                mynrx_sin                           = np.argmax(lst_vars == (myvarx+'_cartesian_sin'))
            if mycircy:
                mynry_cos                           = np.argmax(lst_vars == (myvary+'_cartesian_cos'))
                mynry_sin                           = np.argmax(lst_vars == (myvary+'_cartesian_sin'))

            #correlation coefficient
            if (mycircx and mycircy):
                cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry] = \
                    circular_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_cos, mynry_cos],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_cos, mynry_sin],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_sin, mynry_cos],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_sin, mynry_sin],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_cos, mynrx_sin],
                        cewe_scatter_dct['correlationcoefficientxy'][mynry_cos, mynry_sin],
                    )                
            elif mycircx:
                cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry] = \
                    linear_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_cos, mynry],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_sin, mynry],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx_cos, mynrx_sin],
                    )
            elif mycircy:
                cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry] = \
                    linear_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry_cos],
                        cewe_scatter_dct['correlationcoefficientxy'][mynrx, mynry_sin],
                        cewe_scatter_dct['correlationcoefficientxy'][mynry_cos, mynry_sin],
                    )
            
            #least squares fit parameters
            if mycircx:
                #with respect to the article the following mapping is relevant
                #x1 -> cos_x
                #x2 -> sin_x
                #y -> y
                cewe_scatter_dct['leastsquaresfit_circ_beta0'][mynrx, mynry],\
                cewe_scatter_dct['leastsquaresfit_circ_beta1'][mynrx, mynry],\
                cewe_scatter_dct['leastsquaresfit_circ_beta2'][mynrx, mynry] = \
                    lsf_parameters2(
                        cewe_scatter_dct['meanx'][mynrx_cos, mynry],
                        cewe_scatter_dct['meanx'][mynrx_sin, mynry],
                        cewe_scatter_dct['meany'][mynrx, mynry],
                        cewe_scatter_dct['variancex'][mynrx_cos, mynry],
                        cewe_scatter_dct['variancex'][mynrx_sin, mynry],
                        cewe_scatter_dct['variancey'][mynrx, mynry],
                        cewe_scatter_dct['covariancexy'][mynrx_cos, mynrx_sin],
                        cewe_scatter_dct['covariancexy'][mynrx_cos, mynry],
                        cewe_scatter_dct['covariancexy'][mynrx_sin, mynry],
                    )


    cewe_dct['scatter'] = cewe_scatter_dct



def add_cewe_dct(cewe_dct, cewe_dct_to_add, cewe_opts={}):
    #TBD: check given CEWE dictionaries 
    #...
    
    if len(cewe_dct.keys()) == 0:
        #empty cewe_dct -> copy
        cewe_dct.update(deepcopy(cewe_dct_to_add))
    elif len(cewe_dct_to_add.keys()) == 0:
        #empty cewe_dct_to_add -> nothing 
        pass
    else:

        #add summary statistics, for linear statistics
        for mynr in range(cewe_dct['nvars']):
            n1  = cewe_dct['nsamples'][mynr]
            n2  = cewe_dct_to_add['nsamples'][mynr]
            mycirc  = cewe_dct['circular'][mynr]

            n12     = n1 + n2
            if (n12 != 0):
                if not mycirc:
                    f1      = ((1. * n1) / n12)
                    f2      = ((1. * n2) / n12)
                    
                    #raw moments
                    for i in [1,2,3,4]:
                        cewe_dct['raw_moment_mu_x'+str(i)][mynr] = \
                            (
                            (f1 * cewe_dct['raw_moment_mu_x'+str(i)][mynr])
                            +
                            (f2 * cewe_dct_to_add['raw_moment_mu_x'+str(i)][mynr])
                            )
                        
            cewe_dct['nsamples'][mynr]              = n12        
            
            #histogram
            cewe_dct['histogram_count'][mynr]       = (
                cewe_dct['histogram_count'][mynr] +
                cewe_dct_to_add['histogram_count'][mynr]
                )

        #statistics, pass for linear statistics
        cewe_scatter_dct         = cewe_dct['scatter']
        cewe_scatter_dct_to_add  = cewe_dct_to_add['scatter']

        for mynrx in range(cewe_dct['nvars']):                
            for mynry in range(cewe_dct['nvars']):             
                n1  = cewe_scatter_dct['nsamples'][mynrx, mynry]
                n2  = cewe_scatter_dct_to_add['nsamples'][mynrx, mynry]
                mycircx = int(cewe_dct['circular'][mynrx])
                mycircy = int(cewe_dct['circular'][mynry])
                
                n12 = n1 + n2
                if n12 != 0:
                    f1  = (1. * n1) / n12           
                    f2  = (1. * n2) / n12           

                    if not mycircx:
                        for i in [1,2,3,4]:
                            cewe_scatter_dct['raw_moment_mu_x'+str(i)][mynrx, mynry] = \
                                (
                                (f1 * cewe_scatter_dct['raw_moment_mu_x'+str(i)][mynrx, mynry])
                                +
                                (f2 * cewe_scatter_dct_to_add['raw_moment_mu_x'+str(i)][mynrx, mynry])
                                )
                                            
                    if not mycircy:
                        for i in [1,2,3,4]:
                            cewe_scatter_dct['raw_moment_mu_y'+str(i)][mynrx, mynry] = \
                                (
                                (f1 * cewe_scatter_dct['raw_moment_mu_y'+str(i)][mynrx, mynry])
                                +
                                (f2 * cewe_scatter_dct_to_add['raw_moment_mu_y'+str(i)][mynrx, mynry])
                                )

                    #calculate mixed moments
                    if ((not mycircx) and (not mycircy)):
                        cewe_scatter_dct['mixed_moment_nu'][mynrx, mynry] = \
                            (
                            (f1 * cewe_scatter_dct['mixed_moment_nu'][mynrx, mynry])
                            +
                            (f2 * cewe_scatter_dct_to_add['mixed_moment_nu'][mynrx, mynry])
                            )                    
                                                                            
                cewe_scatter_dct['nsamples'][mynrx, mynry]          = n12
                cewe_scatter_dct['histogram2D_count'][mynrx, mynry]  = (
                    cewe_scatter_dct['histogram2D_count'][mynrx, mynry] +
                    cewe_scatter_dct_to_add['histogram2D_count'][mynrx, mynry]
                    )

        #update
        cewe_udpate(cewe_dct)


def raw_moment(x, k, weights=None):
    if weights == None:
        weights = np.ones(len(x))
    weights = np.array(weights) / np.sum(weights)
    return np.sum(weights*(x**(1.*k)))

def mixed_moment(x, y):
    weights = np.ones(len(x))
    weights = np.array(weights) / np.sum(weights)
    return np.sum(weights * x * y)
    
def circ_mean(x_mean_cos, x_mean_sin):
    return np.arctan2(x_mean_sin, x_mean_cos)

def circ_variance(x_mean_cos, x_mean_sin):
    return -2. * np.log(np.sqrt((x_mean_cos ** 2.) + (x_mean_sin ** 2.)))

def linear_circular_corrcoef(rxc, rxs, rcs):
    return np.sqrt(
        ((rxc ** 2.) + (rxs ** 2.) - (2. * rxc * rxs * rcs)) 
        /
        (1. - (rcs ** 2.))
    )
    
def circular_circular_corrcoef(rCC, rCS, rSC, rSS, r1, r2):
    x = (
            (
                ((rCC**2.) + (rCS ** 2.) + (rSC ** 2.) + (rSS **2.))
                + (2. * ((rCC * rSS) + (rCS * rSC)) * r1 * r2)
                - (2. * ((rCC * rCS) + (rSC * rSS)) * r2)
                - (2. * ((rCC * rSC) + (rCS * rSS)) * r1)
            )
            /
            ((1. - (r1 ** 2.)) * (1. - (r2 ** 2.)))
        )
    return np.sqrt(x) if (x > 0) else np.sqrt(-x)

def lsf_parameters(mux, muy, varx, vary, covxy):
    beta1 = covxy / varx
    beta0 = muy - (beta1 * mux)
    return beta0, beta1
                    
def lsf_parameters2(mux, muy, muZ, varx, vary, varZ, covxy, covxZ, covyZ):
    beta1 = ((covxy * covyZ) - (covxZ * vary)) / ((covxy ** 2.) - (varx * vary))
    beta2 = ((covxy * covxZ) - (covyZ * varx)) / ((covxy ** 2.) - (varx * vary))
    beta0 = muZ - (beta1 * mux) - (beta2 * muy)
    return beta0, beta1, beta2

def ok(x, cewe_opts={}):
    check = (False == (
		np.isinf(x) |
		np.isnan(x) |
		(np.abs(x - cewe_opts['_FillValue']) < 1.e-4)
		))
    return check
    
