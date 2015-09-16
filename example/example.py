#!/usr/bin/env python

import numpy as np

import sys, os; sys.path.append(os.path.expanduser("~/tools/cewe/scripts")); import cewe


#make up some data
def give_random_data():
	n = 1000 + int(10 * np.random.random())
	data = {}
	data['A'] = {
		'data': np.random.normal(size=n),
		'angular': False,
		'llim': -3.,
		'ulim': 3.,
		'plotname': 'A',
		'units': '-',
		'scalefactor': 1.,
		}
	data['B'] = {
		'data': np.random.normal(size=n),
		'angular': True,
		'llim': -3.,
		'ulim': 3.,
		'plotname': 'B',
		'units': '-',
		'scalefactor': 1.,
		}
	data['C'] = {
		'data': data['A']['data'] + data['B']['data'] + 0.1 * np.random.normal(size=n),
		'angular': False,
		'llim': -3.,
		'ulim': 3.,
		'plotname': 'C',
		'units': '-',
		'scalefactor': 1.,
		}
	return data
	

opts = {'bins': 50}

cewe.process(give_random_data(), 'example_stat_set1.h5', opts)
cewe.process(give_random_data(), 'example_stat_set2.h5', opts)
cewe.combine('example.h5', 'example_stat_set1.h5', 'example_stat_set2.h5')

cewe.plot('example.h5', 'A', 'B')
cewe.plot('example.h5', 'A', 'C')



#to be done:
#fix smooth colorbar
#fixed 1:1 saspect ratio
