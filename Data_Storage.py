import nibabel as nib
from nibabel.testing import data_path
import numpy as np
from PIL import Image
import glob
import os
import sys
import random


contrast = 0
contrasts = 4
slice = 0
slices = 155
mode = input("Choose between 'train', 'test', 'labels' or 'validation': ")
if mode == 'validation':
    num_val = int(input("Give the number of images you want to keep as validation: "))
if mode != 'train' and mode != 'test' and mode != 'labels' and mode != 'validation':
    sys.exit('The chosen option is not valid')
format = input("Choose a format, for exemple: '.jpg' ")

def read_data(read_path,filename):
    nii_image = nib.load(read_path + str(filename))
    data_color = nii_image.get_data()
    return data_color


def define_path(write_path, mode):
    if mode == 'train' or mode == 'test':
        slice_name = 'P' + str(patient) + 'C' + str(contrast + 1) + '_' + str(slice + 1) + format
    if mode == 'labels':
        slice_name = 'P' + str(patient) + '_' + str(slice + 1) + format
    slice_path = write_path + slice_name
    return slice_name, slice_path


def convert_grayscale(data_color, mode):
    if mode == 'train' or mode == 'test':
        data_gray = (((data_color[:,:,slice,contrast] - data_color[:,:,slice,contrast].min()) / (data_color[:,:,slice,contrast].max() - data_color[:,:,slice,contrast].min())) * 255.9).astype(np.uint8)
    if mode == 'labels':
        data_gray = (((data_color[:,:,slice] - data_color[:,:,slice].min()) / (data_color[:,:,slice].max() - data_color[:,:,slice].min())) * 255.9).astype(np.uint8)
    return data_gray


def store_data(data_gray, slice_path):
    Image.fromarray(data_gray).save(slice_path)


#When chosing mode = 'validation':
read_path_train = 'tr_nifti/' #path with the nifti images
read_path_labels = 'labels_nifti/' #path with the nifti images
write_path_train = 'tr_png/' #path to save the new images
write_path_validation = 'val_png/' #path to save the new images
write_path_trLabels = 'trLabels_png/' #path to save the new images
write_path_valLabels = 'valLabels_png/' #path to save the new images



if mode == 'train' or mode == 'test':
    if mode == 'train':
        patient = 0
    if mode == 'test':
        patient = 484

    for filename in os.listdir(read_path):
        if filename.startswith("B"):
            patient += 1
            data_color = read_data(read_path,filename)
            for contrast in range(contrasts):
                for slice in range(slices):
                    slice_name, slice_path = define_path(write_path, mode)
                    data_gray = convert_grayscale(data_color, mode)
                    store_data(data_gray, slice_path)


patient = 0
if mode == 'labels':
    for filename in os.listdir(read_path):
        if filename.startswith("B"):
            patient += 1
            data_color = read_data(read_path,filename)
            for slice in range(slices):
                slice_name, slice_path = define_path(write_path, mode)
                data_gray = convert_grayscale(data_color, mode)
                store_data(data_gray, slice_path)



patient == 0
if mode == 'validation':
    val_list = random.sample(range(1,485),num_val)
    val_list.sort() #random patients that will form the validation set
    print(val_list) #We can print the list of the validation patients

    #####
    #training images
    #####

    for filename in os.listdir(read_path_train):
        if filename.startswith("B"):
            patient += 1
            data_color = read_data(read_path_train,filename)

            if patient in val_list: #validation images
                for contrast in range(contrasts):
                    for slice in range(slices):
                        slice_name, slice_path = define_path(write_path_validation, mode='train')
                        data_gray = convert_grayscale(data_color, mode='train')
                        store_data(data_gray, slice_path)

            if patient not in val_list: #training images
                for contrast in range(contrasts):
                    for slice in range(slices):
                        slice_name, slice_path = define_path(write_path_train, mode='train')
                        data_gray = convert_grayscale(data_color, mode='train')
                        store_data(data_gray, slice_path)

    #####
    #labels
    #####
    patient = 0
    for filename in os.listdir(read_path_labels):
        if filename.startswith("B"):
            patient += 1
            data_color = read_data(read_path_labels,filename)

            if patient in val_list: #validation labels
                for slice in range(slices):
                    slice_name, slice_path = define_path(write_path_valLabels, mode='labels')
                    data_gray = convert_grayscale(data_color, mode='labels')
                    store_data(data_gray, slice_path)


            if patient not in val_list: #training labels
                for slice in range(slices):
                    slice_name, slice_path = define_path(write_path_trLabels, mode='labels')
                    data_gray = convert_grayscale(data_color, mode='labels')
                    store_data(data_gray, slice_path)
