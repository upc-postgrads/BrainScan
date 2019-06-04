import nibabel as nib
import os
from PIL import Image
import numpy as np


format = input("write desired output format, for instance: '.jpg'")
if len(format) < 1 : format = '.png'
user=input("Write your username")

#Define folders and variable patient
#These paths are just example paths
read_path ='/Users/'+user+'/Desktop/Task01_BrainTumour/labelsTr/'
write_path ='/Users/'+user+'/Desktop/CleanSlices/labelsTr/'
patient = 0

for filename in os.listdir(read_path):
    if filename.endswith("nii.gz"):
        nii_image = nib.load(read_path+ str(filename))
        data = nii_image.get_data()
        patient += 1
        for slices in range(data.shape[2]):
            slice_name = 'P' + str(patient + 1) + '_' + str(slices + 1) + format
            slice_path = write_path + slice_name
            data_color = (((data[:,:,slices] - data[:,:,slices].min()) / (data[:,:,slices].max() - data[:,:,slices].min())) * 255.9).astype(np.uint8)
            Image.fromarray(data_color).save(slice_path)
            print(slice_name)
