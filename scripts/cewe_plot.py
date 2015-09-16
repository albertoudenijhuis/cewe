#!/usr/bin/env python

import warnings; warnings.simplefilter("ignore")
import numpy as np
from copy import deepcopy
#import sys

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
    
def plot(myfile, var1, var2, opts={}):
    dct = cewe_io.read_stats(myfile)
    scatterdct = dct['scatter']

    #get numbers
    for i in range(len(dct['variables'])):
        if dct['variables'][i] == var1:
            mynrA = i
        if dct['variables'][i] == var2:
            mynrB = i
            
    #make cmap
    mycolors = map(plt.cm.autumn_r ,  np.linspace(0,1., 256.) )
    for i in range(0,len(mycolors)):
        div = 32
        mycolors[i]     = mycolors[ div*int(i/div)   ]
    mycolors[0] = plt.cm.colors.cnames.get('white')
    mycmap = plt.cm.colors.ListedColormap(mycolors, name='mycmap')


    fig = plt.figure(figsize=(5,5))
    fontsize0=14
    matplotlib.rc('xtick', labelsize=fontsize0) 
    matplotlib.rc('ytick', labelsize=fontsize0) 

    ax = fig.add_subplot(1,1,1)
    ax.grid(True)

    xlab = dct['plotname'][mynrA] + " ["+dct['units'][mynrA]+"]"
    ylab = dct['plotname'][mynrB] + " ["+dct['units'][mynrB]+"]"
    ax.set_xlabel(xlab, fontsize=fontsize0)
    ax.set_ylabel(ylab, fontsize=fontsize0)

    bin_count = scatterdct['histogram2D_values'].shape[2]
    #bins = [ np.linspace(0., 1., bin_count + 1),np.linspace(0.,1., bin_count + 1) ] 
    bins = [scatterdct['histogram2D_bounds'][mynrA], scatterdct['histogram2D_bounds'][mynrB]] 
    x = (bins[0][:-1] + bins[0][1:]) / 2.
    y = (bins[1][:-1] + bins[1][1:]) / 2.

    #best geuss for limits
    #xmin, xmax = scatterdct['histogram2D_central_value'][mynrA,0], scatterdct['histogram2D_central_value'][mynrA,-1]
    #ymin, ymax = scatterdct['histogram2D_central_value'][mynrB,0], scatterdct['histogram2D_central_value'][mynrB,-1]
    xmin, xmax = bins[0][0], bins[0][-1]
    ymin, ymax = bins[1][0], bins[1][-1]
        
    this = {}
    this['var'] = deepcopy(scatterdct['histogram2D_values'][mynrA, mynrB])
    
    #obtain xpdf, ypdf
    xpdf = np.zeros(bin_count)
    ypdf = np.zeros(bin_count)              
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
    #note: was plt.imhow ??, vmin=0.,vmax=maxvar, 
    this['cplt'] = ax.imshow(this['var'], cmap=mycmap,extent=(xmin,xmax,ymin,ymax), norm=None,interpolation='nearest', alpha=1.0, zorder=0)

    #plot xpdf, ypdf
    pltstyle1 = {'color':'red' , 'alpha':0.7, 'linewidth':2}
    ax.plot(x, ymin + ((ymax - ymin) * 0.2 * xpdf), **pltstyle1)
    ax.plot(xmin + ((xmax - xmin) * 0.2 * ypdf), y, **pltstyle1)

    #defaulttcks = np.linspace(xmin, xmax, 5)
    #tcks0       = (  var_det.g_av(var1, defaulttcks) - xmin  ) / (xmax - xmin)
    #tcks        = var_det.g_av(var1, defaulttcks)
    #f_str       = lambda x: "{:.1f}".format(x)
    #tkcs_str    = map(f_str, tcks)
    #ax.set_xticks( tcks0  )

    #if var1[-5:] == 'log10':
    #   f_adjust0 = lambda x: "1e"+x
    #   f_adjust = np.vectorize(f_adjust0)
    #   tkcs_str = f_adjust(tkcs_str)
            
    #ax.set_xticklabels(tkcs_str, color='k', fontsize=fontsize0)

    #defaulttcks = np.linspace(ymin, ymax, 5)
    #tcks0       = (  var_det.g_av(var2, defaulttcks) - ymin  ) / (ymax - ymin)
    #tcks        = var_det.g_av(var2, defaulttcks)
    #tkcs_str    = map(f_str, tcks)
    #ax.set_yticks( tcks0  )
    #if var2[-5:] == 'log10':
    #   f_adjust0 = lambda x: "1e"+x
    #   f_adjust = np.vectorize(f_adjust0)
    #   tkcs_str = f_adjust(tkcs_str)

    #ax.set_yticklabels(tkcs_str , color='k', fontsize=fontsize0)

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
        mycolorbar2 = fig.colorbar(this['cplt'], cmap=mycmap, orientation='vertical', shrink=0.55)
        
    for t in mycolorbar2.ax.get_yticklabels():
         t.set_fontsize(fontsize0)

    this['A']       = scatterdct['leastsquaresfit_A'][mynrA, mynrB]
    this['B']       = scatterdct['leastsquaresfit_B'][mynrA, mynrB]
    this['coef']    = scatterdct['correlationcoefficientAB'][mynrA, mynrB]
    
    #gefitte lijn
    rx = np.array([xmin, xmax])
    fit_y= this['A'] + (this['B'] * rx)
    
    ax.plot(rx,fit_y,'r--', linewidth=2)

    #ax.set_xbound(0.,1.)
    #ax.set_ybound(0.,1.)

    #infotext
    sgn_str =  np.where(np.sign(this['B']) == 1, ['+'], ['-'])[0]
    inf0 = "c = {:.2f}\ny = {:.2e} {}{:.2e} x".format(this['coef'], this['A'],sgn_str,np.abs(this['B']))
    txt0 = ax.text(0.04,0.85, inf0 , fontsize=fontsize0, transform = ax.transAxes)
    txt0.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="w")])

    plt.tight_layout()
    
    
    tf = tempfile.NamedTemporaryFile(suffix=".png")
    myname = tf.name 

    ax.set_xbound(xmin, xmax)
    ax.set_ybound(ymin, ymax)


    #plt.savefig(myname)
    plt.savefig("scatterplots/{}_{}.png".format(var1, var2))
    plt.close(fig)

    
