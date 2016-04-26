#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
from copy import deepcopy

import matplotlib; matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
    
import tempfile
import re

import cewe_io

def stripslashes(s):
    r = re.sub(r"\\", "", s)
    r = re.sub(r"/", "", r)
    return r
    
def scatter_density_plot(myfile, myvarX, myvarY, opts={}):
    cewe_dct = cewe_io.read_cewe_hdf5_file(myfile)
    cewe_scatter_dct = cewe_dct['scatter']

    mynrX = np.argmax(np.array(cewe_dct['variables']) == myvarX)
    mynrY = np.argmax(np.array(cewe_dct['variables']) == myvarY)
    mycircX = cewe_dct['circular'][mynrX]
    mycircY = cewe_dct['circular'][mynrY]
    
    if mycircX:
        mynrX_cos   = np.argmax(np.array(cewe_dct['variables']) == (myvarX+'_cartesian_cos'))
        mynrX_sin   = np.argmax(np.array(cewe_dct['variables']) == (myvarX+'_cartesian_sin'))
    if mycircY:
        mynrY_cos   = np.argmax(np.array(cewe_dct['variables']) == (myvarY+'_cartesian_cos'))
        mynrY_sin   = np.argmax(np.array(cewe_dct['variables']) == (myvarY+'_cartesian_sin'))
        
    mycmap = plt.cm.YlOrRd

    fig = plt.figure(figsize=(4,4))

    fontsize0=16
    fontsize1=12
    matplotlib.rc('xtick', labelsize=fontsize0) 
    matplotlib.rc('ytick', labelsize=fontsize0) 

    ax = fig.add_subplot(111)
    ax.grid(True)

    xlab = cewe_dct['plotname'][mynrX] + " ["+cewe_dct['units'][mynrX]+"]"
    ylab = cewe_dct['plotname'][mynrY] + " ["+cewe_dct['units'][mynrY]+"]"
    ax.set_xlabel(xlab, fontsize=fontsize1)
    ax.set_ylabel(ylab, fontsize=fontsize1)

    x = cewe_dct['histogram_central_value'][mynrX]
    y = cewe_dct['histogram_central_value'][mynrY]
    xmin, xmax = cewe_dct['histogram_bounds'][mynrX][0], cewe_dct['histogram_bounds'][mynrX][-1]
    ymin, ymax = cewe_dct['histogram_bounds'][mynrY][0], cewe_dct['histogram_bounds'][mynrY][-1]
        
    this = {}
    this['var'] = deepcopy(cewe_scatter_dct['histogram2D_count'][mynrX, mynrY])
    
    #translate to density
    this['var'] = (1. * this['var']) / np.sum(this['var'])
    
    #obtain xpdf, ypdf
    xpdf = np.zeros(cewe_dct['nbins'])
    ypdf = np.zeros(cewe_dct['nbins'])              
    for i in range(len(this['var'])):
        xpdf[i] = np.sum(this['var'][:,i])
        ypdf[i] = np.sum(this['var'][i,:])
    xpdf /= np.max(xpdf)
    ypdf /= np.max(ypdf)
    
    if 'xdistscaling' in opts.keys():
        for i in range(len(this['var'])):
            ysum = np.sum(this['var'][:,i])
            if ysum > 0.:
                this['var'][:,i] /= ysum

    if 'ydistscaling' in opts.keys():
        for i in range(len(this['var'][0,:])):
            xsum = np.sum(this['var'][i,:])
            if xsum > 0:
                this['var'][i,:] /= xsum
    
    if 'log10' in opts.keys():
        this['var'] = np.log10(this['var'])
        this['var'] = np.where(np.isinf(this['var']) | np.isnan(this['var']), -100., this['var'])
        
    X,Y = np.meshgrid(x, y)

    maxvar = np.max(this['var'])
    if maxvar <= 0.:
        ax.text(0.2, 0.5, "no points", fontsize=fontsize0*2)        
    maxvar = int(np.ceil(maxvar))

    this['var']     = np.flipud(this['var'])        
    this['var2']    = np.where(this['var']  == 0., np.nan, this['var'] ) #do not show zeros
    this['cplt']    = ax.imshow(this['var2'], cmap=mycmap,extent=(xmin,xmax,ymin,ymax), norm=None,interpolation='nearest', alpha=1.0, zorder=0)

    #plot xpdf, ypdf
    pltstyle1 = {'color':'red' , 'alpha':0.7, 'linewidth':2}
    ax.plot(x, ymin + ((ymax - ymin) * 0.2 * xpdf), **pltstyle1)
    ax.plot(xmin + ((xmax - xmin) * 0.2 * ypdf), y, **pltstyle1)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(fontsize1)
                        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(fontsize1)

    if 'log10' in opts.keys():
        lvls    = range(0,maxvar +1)
        f0 = lambda x : "$10^{"+str(int(x))+"}$"
        strvar1 = map(f0, lvls )

        mycolorbar2 = fig.colorbar(this['cplt'], cmap=mycmap, orientation='vertical', ticks=lvls, shrink=0.55)
        mycolorbar2.ax.set_yticklabels (strvar1, color='k', fontsize=fontsize0)
    else:
        mycolorbar2 = fig.colorbar(this['cplt'], cmap=mycmap, orientation='vertical', shrink=0.4)
        
    for t in mycolorbar2.ax.get_yticklabels():
         t.set_fontsize(fontsize1)

    this['coef']    = cewe_scatter_dct['correlationcoefficientXY'][mynrX, mynrY]

    if ((not mycircX) and (not mycircY)):
        #info on correlation
        inf0    = "$r_{XY}$ = "+"{:.2f}".format(this['coef'])

        this['P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY]
        this['P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY]
        
        #gefitte lijn
        rx = np.array([xmin, xmax])
        fit_y= this['P1'] + (this['P2'] * rx)
        ax.plot(rx,fit_y,'r--', linewidth=2, zorder=10)

    if ((mycircX) and (not mycircY)):
        #info on correlation
        inf0    = "$r_{\Theta Y}$ = "+"{:.2f}".format(this['coef'])

        this['P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY]
        this['P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY]
        this['P3']      = cewe_scatter_dct['leastsquaresfit_P3'][mynrX, mynrY]

        #gefitte lijn
        rx      = np.linspace(xmin, xmax, 100)
        fct_x   = 2. * np.pi / (xmax - xmin)
        rx_cos  = np.cos(fct_x * rx)
        rx_sin  = np.sin(fct_x * rx)
        
        fit_y   = this['P1'] + (this['P2'] * rx_cos) + (this['P3'] * rx_sin)
        ax.plot(rx,fit_y,'r--', linewidth=2, zorder=10)

    if ((not mycircX) and (mycircY)):
        #info on correlation
        inf0    = "$r_{X\Theta}$ = "+"{:.2f}".format(this['coef'])

        this['Y_cos_P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY_cos]
        this['Y_cos_P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY_cos]
        this['Y_sin_P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY_sin]
        this['Y_sin_P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY_sin]

        #gefitte lijn
        rx          = np.linspace(xmin, xmax, 100)
        fit_y_cos   = this['Y_cos_P1'] + (this['Y_cos_P2'] * rx) 
        fit_y_sin   = this['Y_sin_P1'] + (this['Y_sin_P2'] * rx) 
        
        fct_y       = 2. * np.pi / (ymax - ymin)
        fit_y       = (1. / fct_y) * np.arctan2(fit_y_sin, fit_y_cos)
        fit_y   = ymin + (fit_y % (ymax - ymin))
        ax.plot(rx,fit_y,'r--', linewidth=2, zorder=10)

    if ((mycircX) and (mycircY)):
        #info on correlation
        inf0    = "$r_{\Theta\Phi}$ = "+"{:.2f}".format(this['coef'])

        this['Y_cos_P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY_cos]
        this['Y_cos_P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY_cos]
        this['Y_cos_P3']      = cewe_scatter_dct['leastsquaresfit_P3'][mynrX, mynrY_cos]
        this['Y_sin_P1']      = cewe_scatter_dct['leastsquaresfit_P1'][mynrX, mynrY_sin]
        this['Y_sin_P2']      = cewe_scatter_dct['leastsquaresfit_P2'][mynrX, mynrY_sin]
        this['Y_sin_P3']      = cewe_scatter_dct['leastsquaresfit_P3'][mynrX, mynrY_sin]

        rx      = np.linspace(xmin, xmax, 100)
        fct_x   = 2. * np.pi / (xmax - xmin)
        rx_cos  = np.cos(fct_x * rx)
        rx_sin  = np.sin(fct_x * rx)

        fit_y_cos   = this['Y_cos_P1'] + (this['Y_cos_P2'] * rx_cos) + (this['Y_cos_P3'] * rx_sin) 
        fit_y_sin   = this['Y_sin_P1'] + (this['Y_sin_P2'] * rx_cos) + (this['Y_sin_P3'] * rx_sin) 

        fct_y       = 2. * np.pi / (ymax - ymin)
        fit_y       = (1. / fct_y) * np.arctan2(fit_y_sin, fit_y_cos)
        fit_y   = ymin + (fit_y % (ymax - ymin))
        ax.plot(rx,fit_y,'r--', linewidth=2, zorder=10)


        #~ sgn_str =  np.where(np.sign(this['P2']) == 1, ['+'], ['-'])[0]
        #~ inf0    = "c = {:.2f}\ny = {:.2e} {}{:.2e} x".format(this['coef'], this['P1'],sgn_str,np.abs(this['P2']))
    

    #infotext
    txt0 = ax.text(0.04,0.90, inf0 , fontsize=fontsize1, transform = ax.transAxes)
    txt0.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])

    ax.set_xbound(xmin, xmax)
    ax.set_ybound(ymin, ymax)

    ax.set_aspect((xmax - xmin) / (ymax - ymin))

    plt.tight_layout()

    if 'givemesrc' in opts.keys():
        tf = tempfile.NamedTemporaryFile(suffix=".png")
        myname = tf.name 
    else:
        myname = "scatterplots/png/{:_<25}_vs_{:_<25}.png".format(myvarY, myvarX)       

    plt.savefig(myname)
    plt.close(fig)

    if 'givemesrc' in opts.keys():
        f = open(myname, "r")
        figsrc = f.read()
        f.close()
        tf.close()
        return figsrc


