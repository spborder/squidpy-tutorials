"""

Using SpatialData to align CODEX dataset from DSA

"""

import os
import sys

import numpy as np

from shapely.geometry import Polygon, Point, box
import tiffslide as ts

import spatialdata as sd
import spatialdata_plot

from math import floor

def main():

    # Reading images (Have to load the whole image as a numpy array in order to initialize )
    histology_img = ts.open_slide('./dsa/15-1 Stitch.svs')
    histo_name = '15-1 Stitch.svs'
    histoId = '6461bbe818a9d4c4175edbcd'
    print(histology_img.dimensions)

    # 
    codex_img = ts.open_slide('./dsa/15-1.tif')
    codex_name = '15-1.tif'
    codexId = '648b46573e6ae3107da0d990'
    print(codex_img.dimensions)







if __name__=='__main__':
    main()












