#!/usr/bin/env python

__readme = \
"""
Part of the CEWE library. See cewe.py for the general details.

For calculations CEWE dictionaries are used.

A CEWE dictionary contains:
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

A CEWE scatter dictionary contains:
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

To create a CEWE ditrionary, a dataset dictionary is used as input.

A dataset ditrionary contains
- A                                 : Variables that are a dictionary
- B                                 : ...
- C                                 : ...

A variable dictionary contains (e.g. 'A')
- angular                           : True/False indicating whether angular variable or not
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

def create_cewe_dct(dataset, input_cewe_opts={}):
    #TBD: check dataset dictionary
    
    cewe_opts = deepcopy(cewe_default_opts)
    cewe_opts.update(input_cewe_opts)
    
    #Fill Values
    _FillValues     = cewe_opts['_FillValue']
    _FillValue      = cewe_opts['_FillValue'][0]

    #take over list of variables
    cewe_dct = {}
    cewe_dct['variables']       = sorted(dataset.keys())

    cewe_dct['nvars']           = len(cewe_dct['variables'])
    cewe_dct['nbins']           = cewe_opts['nbins']

    #list
    cewe_dct['plotname']        = deepcopy(cewe_dct['variables'])
    cewe_dct['scalefactor']     = deepcopy(cewe_dct['variables'])
    cewe_dct['units']           = deepcopy(cewe_dct['variables'])
    
    #numpy lists of integers
    cewe_dct['angular']         = np.zeros(cewe_dct['nvars'], dtype=np.int) + _FillValue
    cewe_dct['nsamples']        = np.zeros(cewe_dct['nvars'], dtype=np.int) + _FillValue
    
    #numpy list of floats
    cewe_dct['histogram_bounds']        = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'] + 1)) + _FillValue
    cewe_dct['histogram_central_value'] = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'])) + _FillValue
    cewe_dct['histogram_count']         = np.zeros((cewe_dct['nvars'], cewe_dct['nbins'])) + _FillValue
    cewe_dct['mean']                    = np.zeros(cewe_dct['nvars']) + _FillValue
    cewe_dct['variance']                = np.zeros(cewe_dct['nvars']) + _FillValue

    for i in range(cewe_dct['nvars']):
        myvar                                   = cewe_dct['variables'][i]
        cewe_dct['angular'][i]                  = 0 if (dataset[myvar]['angular']) else 1
        cewe_dct['histogram_bounds'][i]         = np.linspace(dataset[myvar]['llim'], dataset[myvar]['ulim'], cewe_dct['nbins'] + 1)  
        cewe_dct['histogram_central_value'][i]  = (cewe_dct['histogram_bounds'][i][1:] + cewe_dct['histogram_bounds'][i][:-1]) / 2.
        cewe_dct['plotname'][i]                 = dataset[myvar]['plotname']
        cewe_dct['scalefactor'][i]              = 1. if (dataset[myvar]['scalefactor'] <= 0) else (1. * dataset[myvar]['scalefactor'])
        cewe_dct['units'][i]                    = dataset[myvar]['units']

    for mynr in range(cewe_dct['nvars']): 
        myvar   = cewe_dct['variables'][mynr]
        myok    = ok(dataset[myvar]['samples'], cewe_opts)
        myang   = cewe_dct['angular'][mynr]
        
        cewe_dct['nsamples'][mynr]  = np.sum(myok)
        cewe_dct['angular'][mynr]   = myang

        if cewe_dct['nsamples'][mynr] > 10:
            mysamples = np.compress(myok, dataset[myvar]['samples'])                             
        
            #calculate mean, std, histogram
            cewe_dct['mean'][mynr]              = mean(mysamples, ang=myang)
            cewe_dct['variance'][mynr]          = variance(mysamples, ang=myang)
            myhistogram, dummy                  = np.histogram(mysamples, bins=cewe_dct['histogram_bounds'][i])
            cewe_dct['histogram_count'][mynr]   = deepcopy(myhistogram)
            del mysamples
            
    #Calculate scatter statistics.
    cewe_scatter_dct = {}
    
    sh1 = (cewe_dct['nvars'], cewe_dct['nvars'])
    sh2 = (cewe_dct['nvars'], cewe_dct['nvars'], cewe_dct['nbins'], cewe_dct['nbins'])

    cewe_scatter_dct['nsamples']            = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['meanA']               = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['meanB']               = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['varianceA']           = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['varianceB']           = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['covarianceAB']        = np.zeros(sh1) + _FillValue
    cewe_scatter_dct['histogram2D_count']   = np.zeros(sh2) + _FillValue

    for mynrA in range(cewe_dct['nvars']):                
        myvarA = cewe_dct['variables'][mynrA]
        for mynrB in range(cewe_dct['nvars']):                
            myvarB = cewe_dct['variables'][mynrB]
        
            myokAB =  ok(dataset[myvarA]['samples'], cewe_opts) & ok(dataset[myvarB]['samples'], cewe_opts)
            myangA  = cewe_dct['angular'][mynrA]
            myangB  = cewe_dct['angular'][mynrB]
            
            cewe_scatter_dct['nsamples'][mynrA, mynrB] = np.sum(myokAB)

            if cewe_scatter_dct['nsamples'][mynrA, mynrB] > 10:             
                mydataA = np.compress(myokAB, dataset[myvarA]['samples'])                              
                mydataB = np.compress(myokAB, dataset[myvarB]['samples'])                              
                
                cewe_scatter_dct['meanA'][mynrA, mynrB]          = mean(mydataA, ang=myangA)
                cewe_scatter_dct['meanB'][mynrA, mynrB]          = mean(mydataB, ang=myangB)
                cewe_scatter_dct['varianceA'][mynrA, mynrB]      = variance(mydataA, ang=myangA)
                cewe_scatter_dct['varianceB'][mynrA, mynrB]      = variance(mydataB, ang=myangB)
                cewe_scatter_dct['covarianceAB'][mynrA, mynrB]   = covariance(mydataA, mydataB, myangA, myangB)

                #2D histogram.
                A_llim, A_ulim = dataset[myvarA]['llim'], dataset[myvarA]['ulim']
                B_llim, B_ulim = dataset[myvarB]['llim'], dataset[myvarB]['ulim']

                bins = [ np.linspace(A_llim, A_ulim, cewe_dct['nbins'] + 1), np.linspace(B_llim, B_ulim, cewe_dct['nbins'] + 1) ]

                dummy1, dummy2, dummy3 = np.histogram2d(mydataA, mydataB , bins, normed=False )
                cewe_scatter_dct['histogram2D_count'][mynrA, mynrB] = np.transpose(dummy1) 

    #post calculations
    cewe_scatter_dct['leastsquaresfit_D'] = cewe_scatter_dct['covarianceAB'] / cewe_scatter_dct['varianceA']
    cewe_scatter_dct['leastsquaresfit_C'] = cewe_scatter_dct['meanB'] - (cewe_scatter_dct['leastsquaresfit_D'] * cewe_scatter_dct['meanA'])
    cewe_scatter_dct['correlationcoefficientAB'] = cewe_scatter_dct['covarianceAB'] / np.sqrt(cewe_scatter_dct['varianceA'] * cewe_scatter_dct['varianceB'])
    cewe_dct['scatter'] = cewe_scatter_dct
       
    return cewe_dct             

  
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
              
            #standard
            angular         = cewe_dct['angular'][mynr]
       
            n       = n1 + n2
            mu1     = cewe_dct['mean'][mynr]
            mu2     = cewe_dct_to_add['mean'][mynr]
            mu      = mean((mu1, mu2), (1. * n1/n, 1. * n2/n), angular)
            var1    = cewe_dct['variance'][mynr]
            var2    = cewe_dct_to_add['variance'][mynr]        
            var     = (
                        ((1. * n1) / n) * var1 +
                        ((1. * n1) / n) * (diff(mu1, mu, angular) ** 2.) +
                        ((1. * n2) / n) * var2 +
                        ((1. * n2) / n) * (diff(mu2, mu, angular) ** 2.) 
                        )

            cewe_dct['nsamples'][mynr]              = n            
            cewe_dct['mean'][mynr]                  = mu
            cewe_dct['variance'][mynr]              = var
            cewe_dct['histogram_count'][mynr]       = (
                cewe_dct['histogram_count'][mynr] +
                cewe_dct_to_add['histogram_count'][mynr]
                )

        #scatter statistics
        cewe_scatter_dct         = cewe_dct['scatter']
        cewe_scatter_dct_to_add  = cewe_dct_to_add['scatter']

        for mynrA in range(cewe_dct['nvars']):                
            for mynrB in range(cewe_dct['nvars']):             
                n1  = cewe_scatter_dct['nsamples'][mynrA, mynrB]
                n2  = cewe_scatter_dct_to_add['nsamples'][mynrA, mynrB]
                
                myangA = int(cewe_dct['angular'][mynrA])
                myangB = int(cewe_dct['angular'][mynrB])
                
                if n2 <= 10:
                    #nothing to add
                    continue
                    
                #no results for this variable yet
                if n1 <= 10:
                    cewe_scatter_dct['nsamples'][mynrA, mynrB]          = cewe_scatter_dct_to_add['nsamples'][mynrA, mynrB]
                    cewe_scatter_dct['meanA'][mynrA, mynrB]             = cewe_scatter_dct_to_add['meanA'][mynrA, mynrB]
                    cewe_scatter_dct['meanB'][mynrA, mynrB]             = cewe_scatter_dct_to_add['meanB'][mynrA, mynrB]
                    cewe_scatter_dct['varianceA'][mynrA, mynrB]         = cewe_scatter_dct_to_add['varianceA'][mynrA, mynrB]
                    cewe_scatter_dct['varianceB'][mynrA, mynrB]         = cewe_scatter_dct_to_add['varianceB'][mynrA, mynrB]
                    cewe_scatter_dct['covarianceAB'][mynrA, mynrB]      = cewe_scatter_dct_to_add['covarianceAB'][mynrA, mynrB]
                    cewe_scatter_dct['histogram2D_count'][mynrA, mynrB] = cewe_scatter_dct_to_add['histogram2D_count'][mynrA, mynrB]
                    continue
                    
                n               = n1 + n2
                muA1            = cewe_scatter_dct['meanA'][mynrA, mynrB]
                muA2            = cewe_scatter_dct_to_add['meanA'][mynrA, mynrB]
                muB1            = cewe_scatter_dct['meanB'][mynrA, mynrB]
                muB2            = cewe_scatter_dct_to_add['meanB'][mynrA, mynrB]
                muA             = mean((muA1, muA2), (1. * n1 / n, 1. * n2 / n), myangA)
                muB             = mean((muB1, muB2), (1. * n1 / n, 1. * n2 / n), myangB)
                
                varA1           = cewe_scatter_dct['varianceA'][mynrA, mynrB]
                varA2           = cewe_scatter_dct_to_add['varianceA'][mynrA, mynrB]
                varA            = (1. / n) * ( n1 * varA1 + n1 * (diff(muA1,muA, myangA) ** 2.)
                                        + n2 * varA2 + n2 * (diff(muA2, muA, myangA) ** 2.)  )

                varB1           = cewe_scatter_dct['varianceB'][mynrA, mynrB]
                varB2           = cewe_scatter_dct_to_add['varianceB'][mynrA, mynrB]
                varB            = (1. / n) * ( n1 * varB1 + n1 * (diff(muB1, muB, myangB) ** 2.)
                                        + n2 * varB2 + n2 * (diff(muB2, muB, myangB) ** 2.)  )
                
                #covariance
                covAB1          = cewe_scatter_dct['covarianceAB'][mynrA, mynrB]
                covAB2          = cewe_scatter_dct_to_add['covarianceAB'][mynrA, mynrB]
                covAB =   ( \
                                        ((1. * n1 / n) * covAB1) + ((1. * n1 / n) * diff(muA1, muA, myangA) * diff(muB1, muB, myangB)) + \
                                        ((1. * n2 / n) * covAB2) + ((1. * n2 / n) * diff(muA2, muA, myangA) * diff(muB2, muB, myangB)) \
                          )
                    
                cewe_scatter_dct['nsamples'][mynrA, mynrB]          = n
                cewe_scatter_dct['meanA'][mynrA, mynrB]             = muA
                cewe_scatter_dct['meanB'][mynrA, mynrB]             = muB
                cewe_scatter_dct['varianceA'][mynrA, mynrB]         = varA
                cewe_scatter_dct['varianceB'][mynrA, mynrB]         = varB
                cewe_scatter_dct['covarianceAB'][mynrA, mynrB]      = covAB

                cewe_scatter_dct['histogram2D_count'][mynrA, mynrB]  = (
                    cewe_scatter_dct['histogram2D_count'][mynrA, mynrB] +
                    cewe_scatter_dct_to_add['histogram2D_count'][mynrA, mynrB]
                    )

        #post calculations
        cewe_scatter_dct['leastsquaresfit_D'] = cewe_scatter_dct['covarianceAB'] / cewe_scatter_dct['varianceA']
        cewe_scatter_dct['leastsquaresfit_C'] = cewe_scatter_dct['meanB'] - (cewe_scatter_dct['leastsquaresfit_D'] * cewe_scatter_dct['meanA'])
        cewe_scatter_dct['correlationcoefficientAB'] = cewe_scatter_dct['covarianceAB'] / np.sqrt(cewe_scatter_dct['varianceA'] * cewe_scatter_dct['varianceB'])
        cewe_dct['scatter'] = cewe_scatter_dct
           


def alternative_covariance_calculation(cewe_dct, cewe_opts={}):
    #TBD: check given CEWE dictionaries 
        
    cewe_scatter_dct = cewe_dct['scatter']

    for mynrA in range(cewe_dct['nvars']):                
        for mynrB in range(cewe_dct['nvars']):
            muA         = cewe_scatter_dct['meanA'][mynrA, mynrB]
            muB         = cewe_scatter_dct['meanB'][mynrA, mynrB]
            n           = cewe_scatter_dct['nsamples'][mynrA, mynrB]

            myangA = int(dct['angular'][mynrA])
            myangB = int(dct['angular'][mynrB])  

            cvalueA = cewe_dct['histogram_central_value'][mynrA]
            cvalueB = cewe_dct['histogram_central_value'][mynrB]

            #walk through the grid
            varA = 0.
            varB = 0.
            covAB = 0.
            for iA in range(cewe_dct['bins']):
                for iB in range(cewe_dct['bins']):
                    thisn = cewe_scatter_dct['histogram2D_count'][mynrA, mynrB, iB, iA]
                    
                    varA    += thisn * (diff(cvalueA[iA], muA, myangA)**2.)
                    varB    += thisn * (diff(cvalueB[iB], muB, myangB)**2.)
                    covAB   += thisn * diff(cvalueA[iA], muA, myangA) * diff(cvalueB[iB], muB, myangB)
            
            varA /= n
            varB /= n
            covAB /= n

            lsf_B = covAB / varA
            lsf_A = muB - lsf_B * muA
                        
            #Improve the correction if one of the two variables is angular
            if angularA or angularB:
                varA = 0.
                varB = 0.
                covAB = 0.
                for iA in range(output_cewe_dct['bins']):
                    for iB in range(output_cewe_dct['bins']):
                        thisn = myscatterdct['histogram2d'][mynrA, mynrB, iB, iA]
                        myvalA = cvalueA[iA]
                        myvalB = cvalueB[iB]
                        
                        #clip towards the fitted line
                        if angularA:
                            X = (myvalB - lsf_A) / lsf_B
                            myvalA = X + mydiff(myvalA, X)
                        if angularB:
                            Y = lsf_A + lsf_B * myvalA
                            myvalB = Y + mydiff(myvalB, Y)
                                            
                        varA    += thisn * ((myvalA - muA) ** 2.)
                        varB    += thisn * ((myvalB - muB) ** 2.)
                        covAB   += thisn * (myvalA - muA) * (myvalB - muB)
                
                varA    /= n
                varB    /= n
                covAB   /= n
                
            cewe_scatter_dct['varianceA'][mynrA, mynrB]      = varA
            cewe_scatter_dct['varianceB'][mynrA, mynrB]      = varB
            cewe_scatter_dct['covarianceAB'][mynrA, mynrB]  = covAB

    #post calculations
    cewe_scatter_dct['leastsquaresfit_D']       = cewe_scatter_dct['covarianceAB'] / cewe_scatter_dct['varianceA']
    cewe_scatter_dct['leastsquaresfit_C']       = cewe_scatter_dct['meanB'] - (cewe_scatter_dct['leastsquaresfit_D'] * cewe_scatter_dct['meanA'])
    cewe_scatter_dct['correlationcoefficientAB'] = cewe_scatter_dct['covarianceAB'] / np.sqrt(cewe_scatter_dct['varianceA'] * cewe_scatter_dct['varianceB'])


#Other more trivial functions

def diff(x, y, ang=False):
    if not ang:
        return x - y
    else:
        return ((180. + x - y) % 360.)   - 180.
    
def mean(x, weights=None, ang=False):
    if weights == None:
        weights = np.ones(len(x))
    weights = np.array(weights) / np.sum(weights)
    if not ang:
        return np.sum(weights*x)
    else:
        cos = np.sum(weights*np.cos(np.deg2rad(x)))
        sin = np.sum(weights*np.sin(np.deg2rad(x)))
        return (180. + np.rad2deg(np.arctan2(sin,cos)) % 360.) - 180.
    
def variance(x, ang=False):
    return np.average(diff(x, mean(x, ang=ang))**2.)

def covariance(x1, x2, ang1=False, ang2=False):
    return np.average(diff(x1, mean(x1, ang=ang1))*diff(x2, mean(x2, ang=ang2)))

def corrcoef(x1, x2, ang1 = False, ang2=False):
    return covariance(x1, x2, ang1, ang2) / np.sqrt(variance(x1, ang1) * variance(x2,ang2))

def ok(x, cewe_opts={}):
    check = (False == (np.isinf(x) | np.isnan(x) | (x == cewe_opts['_FillValue'])))
    return check
    
