#!/usr/bin/env python

from cewe_io import *
from cewe_stat import *
from cewe_plot import *

def process(data, filename, opts):
    dct = calc_stats(data, opts)
    write_stats(filename, dct)

def combine(outputfile, inputfile1, inputfile2):
    dct = {}
    dct1 = read_stats(inputfile1)
    dct2 = read_stats(inputfile2)

    update_stats(dct, dct1)
    update_stats(dct, dct2)
    write_stats(outputfile, dct)
