#!/usr/bin/env python


#dct contains:

#variables
#plotname
#units
#scalefactor
#angular
#n
#mean
#variance
#distribution_norm_ppt
#distribution_val
#distribution_cdf

#dct['scatter'] contains:
#n
#meanA
#meanB
#varianceA
#varianceB
#covarianceAB
#correlationcoefficientAB
#leastsquaresfit_A
#leastsquaresfit_B
#histogram2D_bounds
#histogram2D_central_value
#histogram2D_values


import warnings; warnings.simplefilter("ignore")
import numpy as np
from scipy import stats
from copy import deepcopy
from scipy.interpolate import interp1d
#~ from scipy.interpolate import UnivariateSpline




cewe_default_opts = {'bins':100, '_FillValue': -999.9}

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

def ok(x, opts=cewe_default_opts):
    check = (False == (np.isinf(x) | np.isnan(x) | (x == opts['_FillValue'])))
    return check
    
def calc_stats(data, input_opts={}):
    opts = deepcopy(cewe_default_opts)
    opts.update(input_opts)
    
    _FillValue = opts['_FillValue']
    
    dct = {}
    dct['variables'] = sorted(data.keys())
    dct['plotname'] = deepcopy(dct['variables'])
    dct['units'] = deepcopy(dct['variables'])
    dct['scalefactor'] = deepcopy(dct['variables'])
    
    for i in range(len(dct['variables'])):
        myvar = dct['variables'][i]
        dct['plotname'][i] = data[myvar]['plotname']
        dct['units'][i] = data[myvar]['units']
        dct['scalefactor'][i] = data[myvar]['scalefactor']
       
    #step 1: calculate simple statistics
    dct['n']                = np.zeros(len(dct['variables'])) + _FillValue
    dct['angular']          = np.zeros(len(dct['variables'])) + _FillValue
    dct['mean']             = np.zeros(len(dct['variables'])) + _FillValue
    dct['variance']         = np.zeros(len(dct['variables'])) + _FillValue
    dct['distribution_val'] = np.zeros((len(dct['variables']),opts['bins'] + 1)) + _FillValue
    dct['distribution_cdf'] = np.zeros((len(dct['variables']),opts['bins'] + 1)) + _FillValue

    for mynr in range(len(dct['variables'])): 
        myvar   = dct['variables'][mynr]
        myok    = ok(data[myvar]['data'], opts)
        myang   = data[myvar]['angular']
        
        dct['n'][mynr]  = np.sum(myok)
        dct['angular'][mynr] = myang

        if dct['n'][mynr] > 10:
            mydata = np.compress(myok, data[myvar]['data'])                             
        
            #calculate mean, std, scoreatpercentile
            dct['mean'][mynr]           = mean(mydata, ang=myang)
            dct['variance'][mynr]       = variance(mydata, ang=myang)
            [dct['distribution_val'][mynr], dct['distribution_cdf'][mynr]] = give_cdfval(mydata, myang, opts)

            del mydata

    if 'distribution_norm_ppt' in opts.keys():
        dct['distribution_norm_ppt']   = opts['distribution_norm_ppt']
            
    #step 2: calculate scatter statistics.
    scatter_dct = {}
    sh1 = (len(dct['variables']), len(dct['variables']))
    sh2 = (len(dct['variables']), len(dct['variables']), opts['bins'], opts['bins'])
    sh3 = (len(dct['variables']),  opts['bins'] + 1)
    sh4 = (len(dct['variables']),  opts['bins'])
    scatter_dct['n']              = np.zeros(sh1) + _FillValue
    scatter_dct['meanA']          = np.zeros(sh1) + _FillValue
    scatter_dct['meanB']          = np.zeros(sh1) + _FillValue
    scatter_dct['varianceA']      = np.zeros(sh1) + _FillValue
    scatter_dct['varianceB']      = np.zeros(sh1) + _FillValue
    scatter_dct['covarianceAB']   = np.zeros(sh1) + _FillValue

    scatter_dct['histogram2D_bounds']           = np.zeros(sh3) + _FillValue
    scatter_dct['histogram2D_central_value']    = np.zeros(sh4) + _FillValue
    scatter_dct['histogram2D_values']           = np.zeros(sh2) + _FillValue

    for mynrA in range(len(dct['variables'])):                
        myvarA = dct['variables'][mynrA]
        for mynrB in range(len(dct['variables'])):                
            myvarB = dct['variables'][mynrB]
        
            myokAB =  ok(data[myvarA]['data'], opts) & ok(data[myvarB]['data'], opts)
            myangA  = data[myvar]['angular']
            myangB  = data[myvar]['angular']
            
            scatter_dct['n'][mynrA, mynrB] = np.sum(myokAB)

            if scatter_dct['n'][mynrA, mynrB] > 10:             
                mydataA = np.compress(myokAB, data[myvarA]['data'])                              
                mydataB = np.compress(myokAB, data[myvarB]['data'])                              
                
                scatter_dct['meanA'][mynrA, mynrB]          = mean(mydataA, ang=myangA)
                scatter_dct['meanB'][mynrA, mynrB]          = mean(mydataB, ang=myangB)
                scatter_dct['varianceA'][mynrA, mynrB]      = variance(mydataA, ang=myangA)
                scatter_dct['varianceB'][mynrA, mynrB]      = variance(mydataB, ang=myangB)
                scatter_dct['covarianceAB'][mynrA, mynrB]   = covariance(mydataA, mydataB, myangA, myangB)

                #2D histogram.
                Amin, Amax = data[myvarA]['llim'], data[myvarA]['ulim']
                Bmin, Bmax = data[myvarB]['llim'], data[myvarB]['ulim']

                bins = [ np.linspace(Amin, Amax, opts['bins'] + 1),np.linspace(Bmin,Bmax, opts['bins'] + 1) ]
                scatter_dct['histogram2D_bounds'][mynrA] = bins[0]
                scatter_dct['histogram2D_central_value'][mynrA] = (bins[0][1:] + bins[0][:-1]) / 2.

                dummy1, dummy2, dummy3 = np.histogram2d(mydataA, mydataB , bins, normed=False )
                scatter_dct['histogram2D_values'][mynrA, mynrB] = np.transpose(dummy1) 

    #post calculations
    scatter_dct['leastsquaresfit_B'] = scatter_dct['covarianceAB'] / scatter_dct['varianceA']
    scatter_dct['leastsquaresfit_A'] = scatter_dct['meanB'] - (scatter_dct['leastsquaresfit_B'] * scatter_dct['meanA'])
    scatter_dct['correlationcoefficientAB'] = scatter_dct['covarianceAB'] / np.sqrt(scatter_dct['varianceA'] * scatter_dct['varianceB'])
    dct['scatter'] = scatter_dct
       
    return dct             

def give_bin_bounds(mu, si, opts= {}):
    if not ('fstat_values_xx' in opts.keys()):
        myn = opts['bins']
        opts['distribution_norm_ppt'] = np.hstack(( (1./(10. * myn)), np.linspace(1./myn, 1.-(1./myn), myn-1), (1. - (1./ (10. * myn))) ))
        opts['fstat_values_xx'] = stats.norm.ppf(opts['distribution_norm_ppt'])     
    return mu + si * opts['fstat_values_xx']

def give_cdfval(x, ang=False, opts = {}):
    mu = mean(x, ang=ang)
    si = np.sqrt(variance(x, ang=ang))
    myv = give_bin_bounds(mu,si, opts)
    myr = np.zeros(myv.shape)
    
    for i in range(len(myv)):
        myr[i] = np.sum(x <= myv[i]) / (1. * len(x))
        
    return myv, myr


  

def update_stats(dct, extradct, opts={}):
    #stats, case 1: no data in main dicionary yet
    if len(dct.keys()) == 0:
        dct.update(deepcopy(extradct))
    else:
        #stars, case 2: already data, update results
        for mynr in range(len(dct['variables'])): 
            n0  = dct['n'][mynr]
            n1  = extradct['n'][mynr]
            
            if n1 <= 10:
                continue
            
            #no results for this variable yet
            if n0 <= 10:
                dct['n'][mynr]                    = extradct['n'][mynr]
                dct['mean'][mynr]                 = extradct['mean'][mynr]
                dct['variance'][mynr]             = extradct['variance'][mynr] 
                dct['distribution_val'][mynr]     = extradct['distribution_val'][mynr]
                dct['distribution_cdf'][mynr]     = extradct['distribution_cdf'][mynr]
                continue
              
            angular = int(dct['angular'][mynr])
            opts['bins'] = len(dct['distribution_norm_ppt']) - 1
       
            n       = n0 + n1
            mu0     = dct['mean'][mynr]
            mu1     = extradct['mean'][mynr]
            mu      = mean((mu0, mu1), (1. * n0/n, 1. * n1/n), angular)
            var0    = dct['variance'][mynr]
            var1    = extradct['variance'][mynr]        
            var     = (1. / n) * ( n0 * var0 + n0 * (diff(mu0, mu, angular) ** 2.) + n1 * var1 + n1 * (diff(mu1, mu, angular) ** 2.)  )


            distribution_val0   = dct['distribution_val'][mynr]
            distribution_val1   = extradct['distribution_val'][mynr]
            distribution_val    = give_bin_bounds(mu,np.sqrt(var), opts)
            distribution_cdf0   = dct['distribution_cdf'][mynr]
            distribution_cdf1   = extradct['distribution_cdf'][mynr]
            
            if var0 == 0.:
                cdfval_i0   = np.repeat(np.nan, len(myvalues))
            else:
                cdfval_i0   = interp1d(distribution_val0, distribution_cdf0 , kind='linear', bounds_error=False, fill_value=np.nan)(distribution_val)
            
            if var1 == 0.:
                cdfval_i1   = np.repeat(np.nan, len(myvalues))
            else:
                cdfval_i1   = interp1d(distribution_val1, distribution_cdf1 , kind='linear', bounds_error=False, fill_value=np.nan)(distribution_val)
            
            cdfval_i0   = np.where(np.isnan(cdfval_i0) | (cdfval_i0 > 1.) | (cdfval_i0 < 0.), (np.sign(distribution_val - mu0) + 1.)/2. , cdfval_i0)
            cdfval_i1   = np.where(np.isnan(cdfval_i1) | (cdfval_i1 > 1.) | (cdfval_i1 < 0.), (np.sign(distribution_val - mu1) + 1.)/2. , cdfval_i1)

            distribution_cdf      = (1. * n0 * cdfval_i0 + 1. * n1 * cdfval_i1) / n
            distribution_cdf[0]   = 0.
            distribution_cdf[-1]  = 1.

            dct['n'][mynr]                    = n            
            dct['mean'][mynr]                 = mu
            dct['variance'][mynr]             = var
            dct['distribution_val'][mynr]     = distribution_val
            dct['distribution_cdf'][mynr]     = distribution_cdf

        #scatterstats
        scatter_dct         = dct['scatter']
        extrascatter_dct    = extradct['scatter']
            
        #scatterstats, case 1: no data in main dicionary yet
        if len(scatter_dct.keys()) == 0:
            scatter_dct.update(deepcopy(extrascatter_dct))

        #scatterstats, case 2: already data, update results
        for mynrA in range(len(dct['variables'])):                
            for mynrB in range(len(dct['variables'])):             
                n0  = scatter_dct['n'][mynrA, mynrB]
                n1  = extrascatter_dct['n'][mynrA, mynrB]
                
                myangA = int(dct['angular'][mynrA])
                myangB = int(dct['angular'][mynrB])
                
                if n1 <= 10:
                    #nothing to add
                    continue
                    
                #no results for this variable yet
                if n0 <= 10:
                    scatter_dct['n'][mynrA, mynrB]              = extrascatter_dct['n'][mynrA, mynrB]
                    scatter_dct['meanA'][mynrA, mynrB]          = extrascatter_dct['meanA'][mynrA, mynrB]
                    scatter_dct['meanB'][mynrA, mynrB]          = extrascatter_dct['meanB'][mynrA, mynrB]
                    scatter_dct['varianceA'][mynrA, mynrB]      = extrascatter_dct['varianceA'][mynrA, mynrB]
                    scatter_dct['varianceB'][mynrA, mynrB]      = extrascatter_dct['varianceB'][mynrA, mynrB]
                    scatter_dct['covarianceAB'][mynrA, mynrB]   = extrascatter_dct['covarianceAB'][mynrA, mynrB]
                
                n               = n0 + n1
                muA0            = scatter_dct['meanA'][mynrA, mynrB]
                muA1            = extrascatter_dct['meanA'][mynrA, mynrB]
                muB0            = scatter_dct['meanB'][mynrA, mynrB]
                muB1            = extrascatter_dct['meanB'][mynrA, mynrB]
                muA             = mean((muA0, muA1), (1. * n0 / n, 1. * n1 / n), myangA)
                muB             = mean((muB0, muB1), (1. * n0 / n, 1. * n1 / n), myangB)
                
                varA0           = scatter_dct['varianceA'][mynrA, mynrB]
                varA1           = extrascatter_dct['varianceA'][mynrA, mynrB]
                varA            = (1. / n) * ( n0 * varA0 + n0 * (diff(muA0,muA, myangA) ** 2.) + n1 * varA1 + n1 * (diff(muA1, muA, myangA) ** 2.)  )

                varB0           = scatter_dct['varianceB'][mynrA, mynrB]
                varB1           = extrascatter_dct['varianceB'][mynrA, mynrB]
                varB            = (1. / n) * ( n0 * varB0 + n0 * (diff(muB0, muB, myangB) ** 2.) + n1 * varB1 + n1 * (diff(muB1, muB, myangB) ** 2.)  )
                
                #correlation coefficient
                covAB0          = scatter_dct['covarianceAB'][mynrA, mynrB]
                covAB1          = extrascatter_dct['covarianceAB'][mynrA, mynrB]
                covAB =   ( \
                                        ((1. * n0 / n) * covAB0) + ((1. * n0 / n) * diff(muA0, muA, myangA) * diff(muB0, muB, myangB)) + \
                                        ((1. * n1 / n) * covAB1) + ((1. * n1 / n) * diff(muA1, muA, myangA) * diff(muB1, muB, myangB)) \
                          )
                    
                scatter_dct['n'][mynrA, mynrB]           = n
                scatter_dct['meanA'][mynrA, mynrB]          = muA
                scatter_dct['meanB'][mynrA, mynrB]          = muB
                scatter_dct['varianceA'][mynrA, mynrB]      = varA
                scatter_dct['varianceB'][mynrA, mynrB]      = varB
                scatter_dct['covarianceAB'][mynrA, mynrB]  = covAB

                scatter_dct['histogram2D_values'][mynrA, mynrB]     = scatter_dct['histogram2D_values'][mynrA, mynrB] + extrascatter_dct['histogram2D_values'][mynrA, mynrB]
                



        #post calculations
        scatter_dct['leastsquaresfit_B'] = scatter_dct['covarianceAB'] / scatter_dct['varianceA']
        scatter_dct['leastsquaresfit_A'] = scatter_dct['meanB'] - (scatter_dct['leastsquaresfit_B'] * scatter_dct['meanA'])
        scatter_dct['correlationcoefficientAB'] = scatter_dct['covarianceAB'] / np.sqrt(scatter_dct['varianceA'] * scatter_dct['varianceB'])
        dct['scatter'] = scatter_dct
           


#with this function, variance and covariance are calculated from the 2d histogram
#for angular variables, clipping is added to the least sqaures fit, to improve the result.
def stats_from_histogram2d(dct, mynrA, mynrB, opts=cewe_default_opts):
    #alternative way to calculate correlation coefficients
    #alternative way to calculate correlation LSF parameters
    #based on histogram 2d
        
    scatter_dct = dct['scatter']

    for mynrA in range(len(dct['variables'])):                
        for mynrB in range(len(dct['variables'])):
            muA         = scatter_dct['meanA'][mynrA, mynrB]
            muB         = scatter_dct['meanB'][mynrA, mynrB]
            n           = scatter_dct['n'][mynrA, mynrB]

            histogram2D_valuesA = scatter_dct['histogram2D_values'][mynrA]
            histogram2D_valuesB = scatter_dct['histogram2D_values'][mynrB]

            myangA = int(dct['angular'][mynrA])
            myangB = int(dct['angular'][mynrB])  

            #walk through the grid
            varA = 0.
            varB = 0.
            covAB = 0.
            for iA in range(opts['bins']):
                for iB in range(opts['bins']):
                    thisn = myscatterdct['histogram2D_values'][mynrA, mynrB, iB, iA]
                    varA    += thisn * ((cvalueA[iA] - muA) ** 2.)
                    varB    += thisn * ((cvalueB[iB] - muB) ** 2.)
                    covAB   += thisn * (cvalueA[iA] - muA) * (cvalueB[iB] - muB)
            
            varA /= n
            varB /= n
            covAB /= n


            lsf_B = covAB / varA
            lsf_A = muB - lsf_B * muA
            
            
            #now improve correction if one of the two variables is angular
            if angularA or angularB:
                varA = 0.
                varB = 0.
                covAB = 0.
                for iA in range(opts['bins']):
                    for iB in range(opts['bins']):
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
                
                lsf_B = covAB / varA
                lsf_A = muB - lsf_B * muA


            scatter_dct['varianceA'][mynrA, mynrB]      = varA
            scatter_dct['varianceB'][mynrA, mynrB]      = varB
            scatter_dct['covarianceAB'][mynrA, mynrB]  = covAB
            



    #post calculations
    scatter_dct['lsf_B'] = scatter_dct['covarianceAB'] / scatter_dct['varianceA']
    scatter_dct['lsf_A'] = scatter_dct['meanB'] - (scatter_dct['lsf_B'] * scatter_dct['meanA'])
    scatter_dct['correlationcoefficientAB'] = scatter_dct['covarianceAB'] / np.sqrt(scatter_dct['varianceA'] * scatter_dct['varianceB'])
