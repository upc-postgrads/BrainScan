import nibabel as nib
import numpy as np
import os
from PIL import Image

path_images = '/Users/aitorjara/Desktop/Task01_BrainTumour/imagesTr'
path_converted_images = '/Users/aitorjara/Desktop/CleanSlices/imagesTr'
patient = 0
contrast = 0
slices = 0
for filename in os.listdir(path_images):
          if filename.startswith("B"):
              patient += 1
              file_path=path_images+'/'+filename
              nii_img = nib.load(file_path)
              data = nii_img.get_data()
              for contrast in range(4):
                  for slices in range(data.shape[2]):
                      slice_name = 'P'+str(patient)+'C'+str(contrast+1)+'_'+str(slices+1)+'.png'
                      slice_path = path_converted_images+'/'+slice_name
                      data_gray = (((data[:,:,slices,contrast]-data[:,:,slices,contrast].min()) / (data[:,:,slices,contrast].max()-data[:,:,slices,contrast].min()))*255.9).astype(np.uint8)
                      Image.fromarray(data_gray).save(slice_path)
                      print(slice_name)
