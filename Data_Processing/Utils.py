import numpy as np

def cropVolume(img, data = False):
    ''' This helper function removes all the black area around the 2D images
    parameter img: 2D slice
    parameter data: data is retrieved with nibabel but if the .get_data() from nib has seen the same example,
    then data is false'''
