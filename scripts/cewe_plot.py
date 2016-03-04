#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt

import matplotlib.patheffects as PathEffects
import tempfile
import re

import cewe_io

def stripslashes(s):
    r = re.sub(r"\\", "", s)
    r = re.sub(r"/", "", r)
    return r
    
def scatter_density_plot(myfile, var1, var2, opts={}):
    cewe_dct = cewe_io.read_cewe_hdf5_file(myfile)
    cewe_scatter_dct = cewe_dct['scatter']

    for i in range(cewe_dct['nvars']):
        if cewe_dct['variables'][i] == var1:
            mynrA = i
        if cewe_dct['variables'][i] == var2:
            mynrB = i
    
    mycmap = plt.cm.YlOrRd

    fig = plt.figure(figsize=(5,5))

    
    fontsize0=18
    fontsize1=12
    matplotlib.rc('xtick', labelsize=fontsize0) 
    matplotlib.rc('ytick', labelsize=fontsize0) 

    ax = fig.add_subplot(1,1,1)
    ax.grid(True)

    xlab = cewe_dct['plotname'][mynrA] + " ["+cewe_dct['units'][mynrA]+"]"
    ylab = cewe_dct['plotname'][mynrB] + " ["+cewe_dct['units'][mynrB]+"]"
    ax.set_xlabel(xlab, fontsize=fontsize0)
    ax.set_ylabel(ylab, fontsize=fontsize0)

    x = cewe_dct['histogram_central_value'][mynrA]
    y = cewe_dct['histogram_central_value'][mynrB]
    xmin, xmax = cewe_dct['histogram_bounds'][mynrA][0], cewe_dct['histogram_bounds'][mynrA][-1]
    ymin, ymax = cewe_dct['histogram_bounds'][mynrB][0], cewe_dct['histogram_bounds'][mynrB][-1]
        
    this = {}
    this['var'] = deepcopy(cewe_scatter_dct['histogram2D_count'][mynrA, mynrB])
    
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
        raise ValueError, 'maxvar not okee.'
    maxvar = int(np.ceil(maxvar))

    this['var'] = np.flipud(this['var'])        
    this['var2']  = np.where(this['var']  == 0., np.nan, this['var'] ) #do not show zeros
    this['cplt'] = ax.imshow(this['var2'], cmap=mycmap,extent=(xmin,xmax,ymin,ymax), norm=None,interpolation='nearest', alpha=1.0, zorder=0)

    #plot xpdf, ypdf
    pltstyle1 = {'color':'red' , 'alpha':0.7, 'linewidth':2}
    ax.plot(x, ymin + ((ymax - ymin) * 0.2 * xpdf), **pltstyle1)
    ax.plot(xmin + ((xmax - xmin) * 0.2 * ypdf), y, **pltstyle1)

    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(fontsize0)
                        
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(fontsize0)

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

    this['C']       = cewe_scatter_dct['leastsquaresfit_C'][mynrA, mynrB]
    this['D']       = cewe_scatter_dct['leastsquaresfit_D'][mynrA, mynrB]
    this['coef']    = cewe_scatter_dct['correlationcoefficientAB'][mynrA, mynrB]
    
    #gefitte lijn
    rx = np.array([xmin, xmax])
    fit_y= this['C'] + (this['D'] * rx)
    
    ax.plot(rx,fit_y,'r--', linewidth=2, zorder=10)

    #infotext
    sgn_str =  np.where(np.sign(this['D']) == 1, ['+'], ['-'])[0]
    inf0 = "c = {:.2f}\ny = {:.2e} {}{:.2e} x".format(this['coef'], this['C'],sgn_str,np.abs(this['D']))
    txt0 = ax.text(0.04,0.85, inf0 , fontsize=fontsize1, transform = ax.transAxes)
    txt0.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])

    tf = tempfile.NamedTemporaryFile(suffix=".png")
    myname = tf.name 


    ax.set_xbound(xmin, xmax)
    ax.set_ybound(ymin, ymax)

    #plt.gca().set_aspect('equal', 'box')

    plt.tight_layout()

    plt.savefig("scatterplots/png/{:_<25}_vs_{:_<25}.png".format(var2, var1))
    plt.close(fig)

    
