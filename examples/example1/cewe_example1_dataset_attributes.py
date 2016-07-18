#!/usr/bin/env python

import numpy as np

dataset_attributes = {}

dataset_attributes['A'] = {
	'circular': False,
	'llim': -3.,
	'ulim': 3.,
	'plotname': 'A',
	'units': '-',
	'scalefactor': 1.,
	}
dataset_attributes['B'] = {
	'circular': True,
	'llim': 0.,
	'ulim': 2 * np.pi,
	'plotname': 'B',
	'units': 'rad',
	'scalefactor': 1.,
	}
dataset_attributes['C'] = {
	'circular': False,
	'llim': -3.,
	'ulim': 3.,
	'plotname': 'C',
	'units': '-',
	'scalefactor': 1.,
	}
dataset_attributes['D'] = {
	'circular': True,
	'llim': 0.,
	'ulim': 360.,
	'plotname': 'B',
	'units': 'deg',
	'scalefactor': 1.,
	}
