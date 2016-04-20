#!/usr/bin/env python

import numpy as np
import sys, os; sys.path.append(os.path.expanduser("../../scripts")); import cewe
from cewe_example1_dataset_attributes import *
from pprint import pprint

#make up some data
def give_random_dataset():
	
	nsamples = 1000 + int(10 * np.random.random())
	dataset = {}
	dataset.update(dataset_attributes)

	dataset['A']['samples'] = np.random.normal(size=nsamples)
	dataset['B']['samples'] = np.random.normal(size=nsamples) % 360.
	dataset['C']['samples'] = dataset['A']['samples'] + (0.001 * dataset['B']['samples']) + 0.1 * np.random.normal(size=nsamples)
		
	#pprint(dataset)
	
	return dataset
	

cewe.dataset2ceweh5file(give_random_dataset(), 'cewe_example1_dataset1.h5')
cewe.dataset2ceweh5file(give_random_dataset(), 'cewe_example1_dataset2.h5')
cewe.combine_ceweh5files('cewe_example1_dataset1.h5', 'cewe_example1_dataset2.h5', 'cewe_example1_dataset1_and_dataset2.h5')

cewe.scatter_density_plot('cewe_example1_dataset1_and_dataset2.h5', 'A', 'B')
cewe.scatter_density_plot('cewe_example1_dataset1_and_dataset2.h5', 'A', 'C')

