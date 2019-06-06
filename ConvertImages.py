import nibabel as nib
import numpy as np
import os
from PIL import Image
import random


class Convert_Images():
    #The two functions defined in this class apply on our specific dataset.

    def Convert(path_images,path_converted_images,mode):

        #The function takes the nifti images located on path_images and axially slices each 3D image, saving each slice as a
        #grayscale image with uint8 values in the range [0,255] on path_converted_images. The images are saved in png format.
        #This function requires a 'mode' argument so as to specify if we are dealing with 'training', 'testing' or 'label'
        #images. The only difference between the three modes is the way of naming the files: training and label images go
        #from patient 1 to patient 484 whereas testing images go from 485 to 750.

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
            for filename in os.listdir(path_images.replace('/','\\')):
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
            for filename in os.listdir(path_images.replace('/','\\')):
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


    def Validation(path_tr,path_label,path_converted_tr,path_converted_label,path_val,path_val_label,num_val=48):

        #This function converts the training images (located in path_tr) and labels (located in path_label) from nifti to png
        #by slicing each 3D image axially. The png images take uint8 values in [0,255]. Moreover, it also keeps a certain
        #number of training images (num_val) and their corresponding labels for validation.
        #When saving the images: the converted trainig images and their labels are saved in path_converted_tr and
        #path_converted_label respectively; the validation images and their corresponding labels are saved in path_val and
        #path_val_label respectively.

        #The way of naming the images is the same as the previous function

        val_list = random.sample(range(1,485),num_val)
        val_list.sort() #random patients that will form the validation set
        print(val_list) #We can print the list of the validation patients

        patient = 0
        #training images
        for filename_training in os.listdir(path_tr.replace('/','\\')):
            if filename_training.startswith("B"):
                patient += 1
                file_path=path_tr+'/'+filename_training
                nii_img = nib.load(file_path)
                data = nii_img.get_data()
                if patient in val_list: #validation images
                    for contrast in range(4):
                        for slices in range(data.shape[2]):
                            slice_name = 'P'+str(patient)+'C'+str(contrast+1)+'_'+str(slices+1)+'.png'
                            slice_path = path_val+'/'+slice_name
                            data_gray = (((data[:,:,slices,contrast]-data[:,:,slices,contrast].min()) / (data[:,:,slices,contrast].max()-data[:,:,slices,contrast].min()))*255.9).astype(np.uint8)
                            Image.fromarray(data_gray).save(slice_path)
                if patient not in val_list: #training images
                    for contrast in range(4):
                        for slices in range(data.shape[2]):
                            slice_name = 'P'+str(patient)+'C'+str(contrast+1)+'_'+str(slices+1)+'.png'
                            slice_path = path_converted_tr+'/'+slice_name
                            data_gray = (((data[:,:,slices,contrast]-data[:,:,slices,contrast].min()) / (data[:,:,slices,contrast].max()-data[:,:,slices,contrast].min()))*255.9).astype(np.uint8)
                            Image.fromarray(data_gray).save(slice_path)

        patient = 0
        #labels
        for filename_label in os.listdir(path_label.replace('/','\\')):
            if filename_label.startswith("B"):
                patient += 1
                file_path=path_label+'/'+filename_label
                nii_img = nib.load(file_path)
                data = nii_img.get_data()
                if patient in val_list: #validation labels
                    for slices in range(data.shape[2]):
                        slice_name = 'P'+str(patient)+'_'+str(slices+1)+'.png'
                        slice_path = path_val_label+'/'+slice_name
                        data_gray = data[:,:,slices].astype(np.uint8)
                        Image.fromarray(data_gray).save(slice_path)

                if patient not in val_list: #training labels
                    for slices in range(data.shape[2]):
                        slice_name = 'P'+str(patient)+'_'+str(slices+1)+'.png'
                        slice_path = path_converted_label+'/'+slice_name
                        data_gray = data[:,:,slices].astype(np.uint8)
                        Image.fromarray(data_gray).save(slice_path)



#Convert testing images
path_images = 'BrainTumour/imagesTs'
path_converted_images = 'SlicedDataset/imagesTs'
Convert_Images.Convert(path_images,path_converted_images,'testing')

#Convert training and labels keeping 48 random patients for validation
path_tr = 'BrainTumour/imagesTr'
path_label = 'BrainTumour/labelsTr'
path_converted_tr = 'SlicedDataset/imagesTr'
path_converted_label = 'SlicedDataset/labelsTr'
path_val = 'SlicedDataset/imagesVal'
path_val_label = 'SlicedDataset/labelsVal'
Convert_Images.Validation(path_tr,path_label,path_converted_tr,path_converted_label,path_val,path_val_label)
