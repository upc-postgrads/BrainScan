import nibabel as nib
import numpy as np
import os
from PIL import Image

def Convert_Images(path_images,path_converted_images,mode):

    #This function applies on our specific dataset.

    #The function takes the nifti images located on path_images and axially slices each 3D image, saving each slice as a
    #grayscale image with uint8 values in the range [0,255] on path_converted_images. The images are saved in png format.
    #This function requires a 'mode' argument so as to specify if we are dealing with 'training', 'testing' or 'label' images.
    #The only difference between the three modes is the way of naming the files: training and label images go from patient 1 to
    #patient 484 whereas testing images go from 485 to 750.

    #Note: let us see the way of naming the slices by means of an example:
        #-Training/Testing: P1C4_125 -> means Patient 1, Constrast 4, Slice 125.
        #-Labels: P1_148 -> means Patient 1 and slice 148.

    if mode == 'training':
        patient = 0
    elif mode == 'testing':
        patient = 484
    elif mode == 'label':
        patient = 0
    else:
        try:
            patient +=1
        except:
            print('The mode introduced is not valid')


    if mode == 'label':
        for filename in os.listdir(path_images):
            if filename.startswith("B"):
                patient += 1
                file_path=path_images+'/'+filename
                nii_img = nib.load(file_path)
                data = nii_img.get_data()
                for slices in range(data.shape[2]):
                    slice_name = 'P'+str(patient)+'_'+str(slices+1)+'.png'
                    slice_path = path_converted_images+'/'+slice_name
                    data_gray = data[:,:,slices].astype(np.uint8)
                    Image.fromarray(data_gray).save(slice_path)

    else:
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



#Convert training images
path_images = 'BrainTumour/imagesTr'
path_converted_images = 'SlicedDataset/imagesTr'
Convert_Images(path_images,path_converted_images,'training')

#Convert testing images
path_images = 'BrainTumour/imagesTs'
path_converted_images = 'SlicedDataset/imagesTs'
Convert_Images(path_images,path_converted_images,'testing')

#Convert labels
path_images = 'BrainTumour/labelsTr'
path_converted_images = 'SlicedDataset/labelsTr'
Convert_Images(path_images,path_converted_images,'label')
