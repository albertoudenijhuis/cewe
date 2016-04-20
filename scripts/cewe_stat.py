#!/usr/bin/env python

__readme = \
"""
Part of the CEWE library. See cewe.py for the general details.

For calculations CEWE dictionaries are used.

A CEWE dictionary contains:
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

A CEWE scatter dictionary contains:
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
combine_cewe_dct                    : function that combines two CEWE dictionaries
alternative_covariance_calculation  : alternative that calculate covariance from 2D histogram
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
    cewe_dct['mean']                    = np.zeros(cewe_dct['nvars']) + _FillValue
    cewe_dct['variance']                = np.zeros(cewe_dct['nvars']) + _FillValue

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

        if cewe_dct['nsamples'][mynr] > 10:
            mysamples = np.compress(myok, dataset[myvar]['samples'])                             
                
            if not mycirc:
                #calculate mean and variance
                cewe_dct['mean'][mynr]              = mean(mysamples)
                cewe_dct['variance'][mynr]          = variance(mysamples)
                
            #histogram
            myhistogram, dummy                  = np.histogram(mysamples, bins=cewe_dct['histogram_bounds'][mynr])
            cewe_dct['histogram_count'][mynr]   = deepcopy(myhistogram)

            del mysamples

                
            
    #scatter statistics.
    cewe_scatter_dct = {}
    
    sh1 = (cewe_dct['nvars'], cewe_dct['nvars'])
    sh2 = (cewe_dct['nvars'], cewe_dct['nvars'], cewe_dct['nbins'], cewe_dct['nbins'])

    cewe_scatter_dct['nsamples']                    = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['meanX']                       = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['meanY']                       = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['varianceX']                   = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['varianceY']                   = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['covarianceXY']                = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['correlationcoefficientXY']    = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['leastsquaresfit_P1']          = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['leastsquaresfit_P2']          = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['leastsquaresfit_P3']          = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['histogram2D_count']           = np.zeros(sh2) + _FillValue

    #scatter statistics, pass for linear statistics
    for mynrX in range(cewe_dct['nvars']):   
        myvarX = cewe_dct['variables'][mynrX]
        for mynrY in range(cewe_dct['nvars']):  
            myvarY = cewe_dct['variables'][mynrY]
        
            myokXY =  ok(dataset[myvarX]['samples'], cewe_opts) & ok(dataset[myvarY]['samples'], cewe_opts)
            mycircX  = cewe_dct['circular'][mynrX]
            mycircY  = cewe_dct['circular'][mynrY]
          
            cewe_scatter_dct['nsamples'][mynrX, mynrY] = np.sum(myokXY)

            if cewe_scatter_dct['nsamples'][mynrX, mynrY] > 10:             
                mydataX = np.compress(myokXY, dataset[myvarX]['samples'])                              
                mydataY = np.compress(myokXY, dataset[myvarY]['samples'])                              
                
                #mean
                if not mycircX:
                    cewe_scatter_dct['meanX'][mynrX, mynrY] = mean(mydataX)
                if not mycircY:
                    cewe_scatter_dct['meanY'][mynrX, mynrY] = mean(mydataY)

                #variance
                if not mycircX:
                    cewe_scatter_dct['varianceX'][mynrX, mynrY] = variance(mydataX)
                if not mycircY:
                    cewe_scatter_dct['varianceY'][mynrX, mynrY] = variance(mydataY)

                #covariance
                if ((not mycircX) and (not mycircY)):
                    cewe_scatter_dct['covarianceXY'][mynrX, mynrY]  = covariance(mydataX, mydataY)

                #2D histogram.
                X_llim, X_ulim = dataset[myvarX]['llim'], dataset[myvarX]['ulim']
                Y_llim, Y_ulim = dataset[myvarY]['llim'], dataset[myvarY]['ulim']

                bins = [ np.linspace(X_llim, X_ulim, cewe_dct['nbins'] + 1), np.linspace(Y_llim, Y_ulim, cewe_dct['nbins'] + 1) ]

                dummy1, dummy2, dummy3 = np.histogram2d(mydataX, mydataY , bins, normed=False )
                cewe_scatter_dct['histogram2D_count'][mynrX, mynrY] = np.transpose(dummy1) 

    #set scatter dictionary
    cewe_dct['scatter'] = cewe_scatter_dct

    #calculations
    cewe_update_circular(cewe_dct)
    cewe_calc_correlation_and_lsf(cewe_dct)
       
    return cewe_dct             

def cewe_update_circular(cewe_dct):
    cewe_scatter_dct    = cewe_dct['scatter']
    lst_vars            = np.array(cewe_dct['variables'])

    #statistics, pass for circular
    for mynr in range(cewe_dct['nvars']): 
        myvar                                       = cewe_dct['variables'][mynr]
        mycirc                                      = cewe_dct['circular'][mynr]

        if cewe_dct['nsamples'][mynr] > 10:
            if mycirc:
                mynr_cos                            = np.argmax(lst_vars == (myvar+'_cartesian_cos'))
                mynr_sin                            = np.argmax(lst_vars == (myvar+'_cartesian_sin'))
                
                #calculate mean and variance
                cewe_dct['mean'][mynr]              = circ_mean(cewe_dct['mean'][mynr_cos], cewe_dct['mean'][mynr_sin])
                cewe_dct['variance'][mynr]          = circ_variance(cewe_dct['mean'][mynr_cos], cewe_dct['mean'][mynr_sin])

    #scatter statistics, pass for circular
    for mynrX in range(cewe_dct['nvars']):   
        myvarX = cewe_dct['variables'][mynrX]
        for mynrY in range(cewe_dct['nvars']):  
            myvarY = cewe_dct['variables'][mynrY]
        
            mycircX  = cewe_dct['circular'][mynrX]
            mycircY  = cewe_dct['circular'][mynrY]

            if mycircX:
                mynrX_cos                           = np.argmax(lst_vars == (myvarX+'_cartesian_cos'))
                mynrX_sin                           = np.argmax(lst_vars == (myvarX+'_cartesian_sin'))
            if mycircY:
                mynrY_cos                           = np.argmax(lst_vars == (myvarY+'_cartesian_cos'))
                mynrY_sin                           = np.argmax(lst_vars == (myvarY+'_cartesian_sin'))

            if cewe_scatter_dct['nsamples'][mynrX, mynrY] > 10:
                #mean
                if mycircX:
                    cewe_scatter_dct['meanX'][mynrX, mynrY] = circ_mean(cewe_dct['mean'][mynrX_cos], cewe_dct['mean'][mynrX_sin])
                if mycircY:
                    cewe_scatter_dct['meanY'][mynrX, mynrY] = circ_mean(cewe_dct['mean'][mynrY_cos], cewe_dct['mean'][mynrY_sin])

                #variance
                if mycircX:
                    cewe_scatter_dct['varianceX'][mynrX, mynrY] = circ_variance(cewe_dct['mean'][mynrX_cos], cewe_dct['mean'][mynrX_sin])
                if mycircY:
                    cewe_scatter_dct['varianceY'][mynrX, mynrY] = circ_variance(cewe_dct['mean'][mynrY_cos], cewe_dct['mean'][mynrY_sin])                


def cewe_calc_correlation_and_lsf(cewe_dct):
    cewe_scatter_dct    = cewe_dct['scatter']
    lst_vars            = np.array(cewe_dct['variables'])

    #pass for linear
    for mynrX in range(cewe_dct['nvars']):                
        myvarX = cewe_dct['variables'][mynrX]
        for mynrY in range(cewe_dct['nvars']): 
            myvarY = cewe_dct['variables'][mynrY]
        
            mycircX  = cewe_dct['circular'][mynrX]
            mycircY  = cewe_dct['circular'][mynrY]

            #correlation coefficient
            if not (mycircX or mycircY):
                cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY] = cewe_scatter_dct['covarianceXY'][mynrX, mynrY] / np.sqrt(cewe_scatter_dct['varianceX'][mynrX, mynrY] * cewe_scatter_dct['varianceY'][mynrX, mynrY])

            #least squares fit parameters
            if not (mycircX or mycircY):
                cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY], cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY] = \
                    lsf_parameters(
                        cewe_scatter_dct['meanX'][mynrX, mynrY],
                        cewe_scatter_dct['meanY'][mynrX, mynrY],
                        cewe_scatter_dct['varianceX'][mynrX, mynrY],
                        cewe_scatter_dct['varianceY'][mynrX, mynrY],
                        cewe_scatter_dct['covarianceXY'][mynrX, mynrY],
                    )                   
    
    #pass for circular
    for mynrX in range(cewe_dct['nvars']):                
        myvarX = cewe_dct['variables'][mynrX]
        for mynrY in range(cewe_dct['nvars']): 
            myvarY = cewe_dct['variables'][mynrY]
        
            mycircX  = cewe_dct['circular'][mynrX]
            mycircY  = cewe_dct['circular'][mynrY]
            
            if mycircX:
                mynrX_cos                           = np.argmax(lst_vars == (myvarX+'_cartesian_cos'))
                mynrX_sin                           = np.argmax(lst_vars == (myvarX+'_cartesian_sin'))
            if mycircY:
                mynrY_cos                           = np.argmax(lst_vars == (myvarY+'_cartesian_cos'))
                mynrY_sin                           = np.argmax(lst_vars == (myvarY+'_cartesian_sin'))

            #correlation coefficient
            if (mycircX and mycircY):
                cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY] = \
                    circular_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_cos, mynrY_cos],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_cos, mynrY_sin],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_sin, mynrY_cos],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_sin, mynrY_sin],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_cos, mynrX_sin],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrY_cos, mynrY_sin],
                    )                
            elif mycircX:
                cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY] = \
                    linear_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_cos, mynrY],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_sin, mynrY],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX_cos, mynrX_sin],
                    )
            elif mycircY:
                cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY] = \
                    linear_circular_corrcoef(
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY_cos],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY_sin],
                        cewe_scatter_dct['correlationcoefficientXY'][mynrY_cos, mynrY_sin],
                    )
            
            #least squares fit parameters
            if mycircX:
                #with respect to the article the following mapping is relevant
                #X -> cos_X
                #Y -> sin_X
                #Z -> Y
                cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY],\
                cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY],\
                cewe_scatter_dct['leastsquaresfit_P3'][mynrX, mynrY] = \
                    lsf_parameters2(
                        cewe_scatter_dct['meanX'][mynrX_cos, mynrY],
                        cewe_scatter_dct['meanX'][mynrX_sin, mynrY],
                        cewe_scatter_dct['meanY'][mynrX, mynrY],
                        cewe_scatter_dct['varianceX'][mynrX_cos, mynrY],
                        cewe_scatter_dct['varianceX'][mynrX_sin, mynrY],
                        cewe_scatter_dct['varianceY'][mynrX, mynrY],
                        cewe_scatter_dct['covarianceXY'][mynrX_cos, mynrX_sin],
                        cewe_scatter_dct['covarianceXY'][mynrX_cos, mynrY],
                        cewe_scatter_dct['covarianceXY'][mynrX_sin, mynrY],
                    )

def add_cewe_dct(cewe_dct, cewe_dct_to_add, cewe_opts={}):
    #TBD: check given CEWE dictionaries 

    if len(cewe_dct.keys()) == 0:
        #empty cewe_dct -> copy
        cewe_dct.update(deepcopy(cewe_dct_to_add))
    elif len(cewe_dct_to_add.keys()) == 0:
        #empty cewe_dct_to_add -> nothing 
        pass
    else:

        for mynr in range(cewe_dct['nvars']):
            n1  = cewe_dct['nsamples'][mynr]
            n2  = cewe_dct_to_add['nsamples'][mynr]
            
            if n2 <= 10:
                continue
            
            #no results for this variable yet -> copy
            if n1 <= 10:
                cewe_dct['nsamples'][mynr]             = cewe_dct_to_add['nsamples'][mynr]
                cewe_dct['mean'][mynr]                 = cewe_dct_to_add['mean'][mynr]
                cewe_dct['variance'][mynr]             = cewe_dct_to_add['variance'][mynr] 
                cewe_dct['histogram_count'][mynr]      = cewe_dct_to_add['histogram_count'][mynr] 
                continue
              
            mycirc  = cewe_dct['circular'][mynr]
       
            #statistics, pass for linear statistics
            if not mycirc:
                n12     = n1 + n2
                mu1     = cewe_dct['mean'][mynr]
                mu2     = cewe_dct_to_add['mean'][mynr]
                mu12    = mean((mu1, mu2), (1. * n1/n12, 1. * n2/n12))
                var1    = cewe_dct['variance'][mynr]
                var2    = cewe_dct_to_add['variance'][mynr]        
                var12   = (
                            ((1. * n1) / n12) * var1 +
                            ((1. * n1) / n12) * ((mu1 - mu12) ** 2.) +
                            ((1. * n2) / n12) * var2 +
                            ((1. * n2) / n12) * ((mu2 - mu12) ** 2.)
                            )

                cewe_dct['nsamples'][mynr]              = n12        
                cewe_dct['mean'][mynr]                  = mu12
                cewe_dct['variance'][mynr]              = var12
            
            #histogram
            cewe_dct['histogram_count'][mynr]       = (
                cewe_dct['histogram_count'][mynr] +
                cewe_dct_to_add['histogram_count'][mynr]
                )

        #statistics, pass for linear statistics
        cewe_scatter_dct         = cewe_dct['scatter']
        cewe_scatter_dct_to_add  = cewe_dct_to_add['scatter']

        for mynrX in range(cewe_dct['nvars']):                
            for mynrY in range(cewe_dct['nvars']):             
                n1  = cewe_scatter_dct['nsamples'][mynrX, mynrY]
                n2  = cewe_scatter_dct_to_add['nsamples'][mynrX, mynrY]
                
                mycircX = int(cewe_dct['circular'][mynrX])
                mycircY = int(cewe_dct['circular'][mynrY])
                
                if n2 <= 10:
                    #nothing to add
                    continue
                    
                #no results for this variable yet
                if n1 <= 10:
                    cewe_scatter_dct['nsamples'][mynrX, mynrY]          = cewe_scatter_dct_to_add['nsamples'][mynrX, mynrY]
                    cewe_scatter_dct['meanX'][mynrX, mynrY]             = cewe_scatter_dct_to_add['meanX'][mynrX, mynrY]
                    cewe_scatter_dct['meanY'][mynrX, mynrY]             = cewe_scatter_dct_to_add['meanY'][mynrX, mynrY]
                    cewe_scatter_dct['varianceX'][mynrX, mynrY]         = cewe_scatter_dct_to_add['varianceX'][mynrX, mynrY]
                    cewe_scatter_dct['varianceY'][mynrX, mynrY]         = cewe_scatter_dct_to_add['varianceY'][mynrX, mynrY]
                    cewe_scatter_dct['covarianceXY'][mynrX, mynrY]      = cewe_scatter_dct_to_add['covarianceXY'][mynrX, mynrY]
                    cewe_scatter_dct['histogram2D_count'][mynrX, mynrY] = cewe_scatter_dct_to_add['histogram2D_count'][mynrX, mynrY]
                    continue
                    
                n12             = n1 + n2
                muX1            = cewe_scatter_dct['meanX'][mynrX, mynrY]
                muX2            = cewe_scatter_dct_to_add['meanX'][mynrX, mynrY]
                muY1            = cewe_scatter_dct['meanY'][mynrX, mynrY]
                muY2            = cewe_scatter_dct_to_add['meanY'][mynrX, mynrY]
                muX12           = mean((muX1, muX2), (1. * n1 / n12, 1. * n2 / n12))
                muY12           = mean((muY1, muY2), (1. * n1 / n12, 1. * n2 / n12))
                                
                varX1           = cewe_scatter_dct['varianceX'][mynrX, mynrY]
                varX2           = cewe_scatter_dct_to_add['varianceX'][mynrX, mynrY]
                varX12          = (1. / n12) * (
                                    (n1 * varX1)
                                    + (n1 * ((muX1 - muX12) ** 2.))
                                    + (n2 * varX2)
                                    + (n2 * ((muX2 - muX12) ** 2.))
                                    )

                varY1           = cewe_scatter_dct['varianceY'][mynrX, mynrY]
                varY2           = cewe_scatter_dct_to_add['varianceY'][mynrX, mynrY]
                varY12          = (1. / n12) * (
                                    (n1 * varY1)
                                    + (n1 * ((muY1 - muY12) ** 2.))
                                    + (n2 * varY2)
                                    + (n2 * ((muY2 - muY12) ** 2.))
                                    )
                
                #covariance
                covXY1          = cewe_scatter_dct['covarianceXY'][mynrX, mynrY]
                covXY2          = cewe_scatter_dct_to_add['covarianceXY'][mynrX, mynrY]
                covXY12 = (
                            ((1. * n1 / n12) * covXY1)
                            + ((1. * n1 / n12) * (muX1 - muX12) * (muY1 - muY12)) 
                            + ((1. * n2 / n12) * covXY2)
                            + ((1. * n2 / n12) * (muX2 - muX12) * (muY2 - muY12))
                          )
                                    
                cewe_scatter_dct['nsamples'][mynrX, mynrY]          = n12
                cewe_scatter_dct['meanX'][mynrX, mynrY]             = muX12
                cewe_scatter_dct['meanY'][mynrX, mynrY]             = muY12
                cewe_scatter_dct['varianceX'][mynrX, mynrY]         = varX12
                cewe_scatter_dct['varianceY'][mynrX, mynrY]         = varY12
                cewe_scatter_dct['covarianceXY'][mynrX, mynrY]      = covXY12

                cewe_scatter_dct['histogram2D_count'][mynrX, mynrY]  = (
                    cewe_scatter_dct['histogram2D_count'][mynrX, mynrY] +
                    cewe_scatter_dct_to_add['histogram2D_count'][mynrX, mynrY]
                    )

        #post calculations
        cewe_update_circular(cewe_dct)
        cewe_calc_correlation_and_lsf(cewe_dct)

#~ def alternative_covariance_calculation(cewe_dct, cewe_opts={}):
    #~ #TBD: check given CEWE dictionaries 
        
    #~ cewe_scatter_dct = cewe_dct['scatter']

    #~ for mynrA in range(cewe_dct['nvars']):                
        #~ for mynrB in range(cewe_dct['nvars']):
            #~ muA         = cewe_scatter_dct['meanA'][mynrA, mynrB]
            #~ muB         = cewe_scatter_dct['meanB'][mynrA, mynrB]
            #~ n           = cewe_scatter_dct['nsamples'][mynrA, mynrB]

            #~ myangA = int(dct['angular'][mynrA])
            #~ myangB = int(dct['angular'][mynrB])  

            #~ cvalueA = cewe_dct['histogram_central_value'][mynrA]
            #~ cvalueB = cewe_dct['histogram_central_value'][mynrB]

            #~ #walk through the grid
            #~ varA = 0.
            #~ varB = 0.
            #~ covAB = 0.
            #~ for iA in range(cewe_dct['bins']):
                #~ for iB in range(cewe_dct['bins']):
                    #~ thisn = cewe_scatter_dct['histogram2D_count'][mynrA, mynrB, iB, iA]
                    
                    #~ varA    += thisn * (diff(cvalueA[iA], muA, myangA)**2.)
                    #~ varB    += thisn * (diff(cvalueB[iB], muB, myangB)**2.)
                    #~ covAB   += thisn * diff(cvalueA[iA], muA, myangA) * diff(cvalueB[iB], muB, myangB)
            
            #~ varA /= n
            #~ varB /= n
            #~ covAB /= n

            #~ lsf_B = covAB / varA
            #~ lsf_A = muB - lsf_B * muA
                        
            #~ #Improve the correction if one of the two variables is angular
            #~ if angularA or angularB:
                #~ varA = 0.
                #~ varB = 0.
                #~ covAB = 0.
                #~ for iA in range(output_cewe_dct['bins']):
                    #~ for iB in range(output_cewe_dct['bins']):
                        #~ thisn = myscatterdct['histogram2d'][mynrA, mynrB, iB, iA]
                        #~ myvalA = cvalueA[iA]
                        #~ myvalB = cvalueB[iB]
                        
                        #~ #clip towards the fitted line
                        #~ if angularA:
                            #~ X = (myvalB - lsf_A) / lsf_B
                            #~ myvalA = X + mydiff(myvalA, X)
                        #~ if angularB:
                            #~ Y = lsf_A + lsf_B * myvalA
                            #~ myvalB = Y + mydiff(myvalB, Y)
                                            
                        #~ varA    += thisn * ((myvalA - muA) ** 2.)
                        #~ varB    += thisn * ((myvalB - muB) ** 2.)
                        #~ covAB   += thisn * (myvalA - muA) * (myvalB - muB)
                
                #~ varA    /= n
                #~ varB    /= n
                #~ covAB   /= n
                
            #~ cewe_scatter_dct['varianceA'][mynrA, mynrB]      = varA
            #~ cewe_scatter_dct['varianceB'][mynrA, mynrB]      = varB
            #~ cewe_scatter_dct['covarianceAB'][mynrA, mynrB]  = covAB

    #~ #post calculations
    #~ cewe_scatter_dct['leastsquaresfit_D']       = cewe_scatter_dct['covarianceAB'] / cewe_scatter_dct['varianceA']
    #~ cewe_scatter_dct['leastsquaresfit_C']       = cewe_scatter_dct['meanB'] - (cewe_scatter_dct['leastsquaresfit_D'] * cewe_scatter_dct['meanA'])
    #~ cewe_scatter_dct['correlationcoefficientAB'] = cewe_scatter_dct['covarianceAB'] / np.sqrt(cewe_scatter_dct['varianceA'] * cewe_scatter_dct['varianceB'])


#~ #Other more trivial functions
#~ def diff(x, y, ang=False):
    #~ if not ang:
        #~ return x - y
    #~ else:
        #~ return ((180. + x - y) % 360.)   - 180.
    
def mean(x, weights=None):
    if weights == None:
        weights = np.ones(len(x))
    weights = np.array(weights) / np.sum(weights)
    return np.sum(weights*x)

def circ_mean(x_mean_cos, x_mean_sin):
    return np.arctan2(x_mean_sin, x_mean_cos)
    
def variance(x):
    return np.average((x - mean(x))**2.)

def circ_variance(x_mean_cos, x_mean_sin):
    return -2. * np.log(np.sqrt((x_mean_cos ** 2.) + (x_mean_sin ** 2.)))

def covariance(x1, x2):
    return np.average((x1 - mean(x1)) * (x2 - mean(x2)))

def corrcoef(x1, x2):
    return covariance(x1, x2) / np.sqrt(variance(x1) * variance(x2))

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

def lsf_parameters(muX, muY, varX, varY, covXY):
    p2 = covXY / varX
    p1 = muY - (p2 * muX)
    return p1, p2
                    
def lsf_parameters2(muX, muY, muZ, varX, varY, varZ, covXY, covXZ, covYZ):
    p2 = ((covXY * covYZ) - (covXZ * varY)) / ((covXY ** 2.) - (varX * varY))
    p3 = ((covXY * covXZ) - (covYZ * varX)) / ((covXY ** 2.) - (varX * varY))
    p1 = muZ - (p2 * muX) - (p3 * muY)
    return p1, p2, p3

def ok(x, cewe_opts={}):
    check = (False == (np.isinf(x) | np.isnan(x) | (x == cewe_opts['_FillValue'])))
    return check
    
