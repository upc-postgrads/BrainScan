import nibabel as nib
from nibabel.testing import data_path
import numpy as np
from PIL import Image
import glob
import os

patient = 0
contrast = 0
contrasts = 4
slice = 0
slices = 155
read_path = '/Users/aitorjara/Desktop/Task01_BrainTumour/imagesTr/' #Indicar path
write_path = '/Users/aitorjara/Desktop/CleanSlices/imagesTr/'       #Indicar path
mode = 'train'#input("Indica 'train', 'test' o 'labels': ")
format = '.png' #input("Indica format, exemple: '.jpg' ")

def read_data(read_path):
    for filename in os.listdir(read_path):
        if filename.startswith("B"):
            nii_image = nib.load(read_path + str(filename))
            data_color = nii_image.get_data()
            return data_color

def define_path(write_path, mode):
    if mode == 'train' or 'test':
        slice_name = 'P' + str(patient) + 'C' + str(contrast + 1) + '_' + str(slice + 1) + format
    if mode == 'labels':
        slice_name = 'P' + str(patient) + '_' + str(slice + 1) + format
    slice_path = write_path + slice_name
    return slice_name, slice_path

def convert_grayscale(data_color, mode):
    if mode == 'train' or 'test':
        data_gray = (((data_color[:,:,slice,contrast] - data_color[:,:,slice,contrast].min()) / (data_color[:,:,slice,contrast].max() - data_color[:,:,slice,contrast].min())) * 255.9).astype(np.uint8)
    if mode == 'labels':
        data_gray = (((data_color[:,:,slice] - data_color[:,:,slice].min()) / (data_color[:,:,slice].max() - data_color[:,:,slice].min())) * 255.9).astype(np.uint8)
    return data_gray

def store_data(data_gray, slice_path):
    Image.fromarray(data_gray).save(slice_path)

patient += 1
if mode == 'train' or 'test':
    for contrast in range(contrasts):
        for slice in range(slices):
           slice_name, slice_path = define_path(write_path, mode)
           data_color = read_data(read_path)
           data_gray = convert_grayscale(data_color, mode)
           store_data(data_gray, slice_path)
           print(slice_name)
if mode == 'labels':
    for slice in range(slices):
       slice_name, slice_path = define_path(write_path, mode)
       data_color = read_data(read_path)
       data_gray = convert_grayscale(data_color, mode)
       store_data(data_gray, slice_path)
       print(slice_name)
