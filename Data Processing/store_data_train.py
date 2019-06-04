import nibabel as nib
import os
from PIL import Image
import numpy as np


format = input("write desired output format, for instance: '.jpg'")
if len(format) < 1 : format = '.png'
user = input("Write your username")

#Define folders and variable patient
#These paths are just example paths
read_path ='/Users/+ user +/Desktop/Task01_BrainTumour/imagesTr/'
write_path ='/Users/+ user +/Desktop/CleanSlices/imagesTr/'
patient = 0

#Iterate through the items in the folder, read data, get data, slice it, decide a name convention and save it
for filename in os.listdir(read_path):
    if filename.endswith("nii.gz"):
        nii_image = nib.load(read_path+ str(filename))
        data = nii_image.get_data()
        patient += 1
        for contrast in range(data.shape[3]):
            for slices in range(data.shape[2]):
                slice_name = 'P' + str(patient) + 'C' + str(contrast + 1) + '_' + str(slices + 1) + format
                slice_path = write_path + slice_name
                data_gray = (((data[:,:,slices,contrast] - data[:,:,slices,contrast].min()) / (data[:,:,slices,contrast].max() - data[:,:,slices,contrast].min())) * 255.9).astype(np.uint8)
                Image.fromarray(data_gray).save(slice_path)
                print(slice_name)
