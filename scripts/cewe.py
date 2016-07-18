#!/usr/bin/env python

__readme = \
'''
CEWE
===============
Comparison of everything with everything (CEWE): A tool for comparison of big data

Author
======
Albert Oude Nijhuis <albertoudenijhuis@gmail.com>

Institute
=========
Delft University of Technology

Date
====
March 3th, 2016

Version
=======
1.0

Files:
=======
cewe.py			: the main file
cewe_stat.py	: library with functions regarding statistics
cewe_io.py		: libary with functions for hdf5 input / output
cewe_plot.py	: libary with plotting functions

Project
=======
EU FP7 program, the UFO project

Acknowledgement and citation
============================
Whenever this Python class is used for publication of scientific results,
the code writer should be informed, acknowledged and referenced.

If you have any suggestions for improvements or amendments, please inform the author of this class.

Typical usage
=============
- See the example.

Testing
=======
- See the example.

Revision History
================
March 3th, 2016
- Code updated such that it can be used more easily.

July 15th, 2016
- Code updated. Raw moments and a mixed moment are now used in the calculations to add skewness and kurtosis conveniently.

'''


from cewe_io import *
from cewe_stat import *
from cewe_plot import *

def dataset2ceweh5file(dataset, cewe_hdf5_filename, cewe_opts={}):
    cewe_dct = create_cewe_dct(dataset, cewe_opts)
    write_cewe_hdf5_file(cewe_hdf5_filename, cewe_dct)

def combine_ceweh5files(cewe_hdf5_input_filename1, cewe_hdf5_input_filename2, cewe_hdf5_output_filename):
    dct = {}
    cewe_dct1 = read_cewe_hdf5_file(cewe_hdf5_input_filename1)
    cewe_dct2 = read_cewe_hdf5_file(cewe_hdf5_input_filename2)

    add_cewe_dct(cewe_dct1, cewe_dct2)
    write_cewe_hdf5_file(cewe_hdf5_output_filename, cewe_dct1)
