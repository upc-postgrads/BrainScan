import nibabel as nib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from utils import utils
import random

class GenerateTFRedord:
    """Class for generate TFREcords from NII volumes.
       The resulting structure of the TFRecord will be:
                       'image': _bytes_feature(frames.tostring()),
                       'label': _bytes_feature(label.tostring()),
                       'PatientID': _int64_feature(int(self.get_patient_id(inputFileVolume))),
                       'Slide': _int64_feature(i),
                       'height': _int64_feature(self.HEIGHT),
                       'width': _int64_feature(self.WIDTH),
                       'depth': _int64_feature(self.DEPTH),
    :param image_path_tr: Location of the training volumes.
    :param label_path_tr: Location of the label volumes.
    :param image_path_ts: Location of the testing volumes.
    :param output_dir: Location where the generated files will be placed.
                       For training the resulting path will be: output_dir/Trainig.
                       For validation files the resulting path will be: output_dir/Validation.
                       For test files the resulting path will be: output_dir/Test.
    :param number_of_volumes_to_process: The number of volumes to process. both for training, for testing.
    :param percent_for_validation: Percentage of the resulting images that will be reserved for validation.
                                   Ej. If value is 10, then the 10% (random) of the TFRecords will be reserved for validation.
    :param fifty_percent_of_labeled:True/False
                        If True, then 50% of labeled images and 50% of non labeled are returned.
                        If False, then all images are used, it does not matter if labeled or not  
    """

    OUTPUT_FILE_TYPE = "jpg" # "jpg", "png"
    LABELS_INDEX = [0,1,2,3]
    FRAMES_INDEX = [0,1,2,3]
    SLICE_START =0
    SLICE_END = 154
    PATIENT_START_TRAINING =1
    PATIENT_END_TRAINING  = 484
    PATIENT_START_TEST =485
    PATIENT_END_TEST  = 750
    HEIGHT=240
    WIDTH=240
    DEPTH=4
    

    def __init__(self, image_path_tr,label_path_tr,image_path_ts,output_dir,number_of_volumes_to_process,percent_for_validation=5,fifty_percent_of_labeled=True):
        self.imagePathTR = image_path_tr
        self.labelPathTR = label_path_tr
        self.imagePathTS=image_path_ts
        self.outputDir=output_dir
        self.outputImagePathTR=os.path.join(self.outputDir,"Training")
        self.outputImagePathVL=os.path.join(self.outputDir,"Validation")
        self.outputimagePathTS=os.path.join(self.outputDir,"Test")
        self.number_of_volumes_to_process=number_of_volumes_to_process
        self.percent_for_validation=percent_for_validation
        self.fifty_percent_of_labeled=fifty_percent_of_labeled

    def get_patient_id(self,fileName):
        fileName=os.path.basename(fileName)
        PatientID = fileName.split('.')[0]
        PatientID = PatientID[-3:]
        PatientID = str(int(PatientID))
        return PatientID

    def is_valid_image(self,img):
        value = True
        if img.min()==img.max() and (img.min()==0 or img.min()==255):
            value=False

        return value

    def get_gray_scale(self,data):
        data = ((np.subtract(data[:,:],data[:,:].min(), dtype=np.float32) / np.subtract(data[:,:].max(),data[:,:].min(), dtype=np.float32))*255.9).astype(np.uint8)
        return data


    def generate_tfrecord_from_patient(self,inputFileVolume,InputFileLabel, is_train_tfrecord, is_val_tfrecord, is_test_tfrecord,):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        outputFile=os.path.basename(inputFileVolume).replace("nii.gz", "tfrecords")
        
        if is_train_tfrecord:
            outputFile=os.path.join(self.outputImagePathTR,outputFile)
        elif is_val_tfrecord:
            outputFile=os.path.join(self.outputImagePathVL, outputFile)
        elif is_test_tfrecord:
            outputFile=os.path.join(self.outputimagePathTS,outputFile)
            
        
        writer = tf.python_io.TFRecordWriter(outputFile)
       

        nii_vol = nib.load(inputFileVolume)
        dataVol = nii_vol.get_data()

        if not is_test_tfrecord:
            nii_label = nib.load(InputFileLabel)
            datalabel = nii_label.get_data()

        frame0=dataVol[:,:,:,0]
        frame1=dataVol[:,:,:,1]
        frame2=dataVol[:,:,:,2]
        frame3=dataVol[:,:,:,3]
        
        #Only are valid the images with some part of brain
        images_with_label = []
        images_without_label = []
        for i in range(self.SLICE_START, self.SLICE_END+1):

            frame_0_gray_scale= self.get_gray_scale(frame0[:, :, i])
            frame_1_gray_scale= self.get_gray_scale(frame1[:, :, i])
            frame_2_gray_scale= self.get_gray_scale(frame2[:, :, i])
            frame_3_gray_scale= self.get_gray_scale(frame3[:, :, i])
            
            if not is_test_tfrecord:
                label=datalabel[:,:,i].astype(np.uint8)
                label = np.array(label)
            
            ImageEmpty=not self.is_valid_image(frame_0_gray_scale) or not self.is_valid_image(frame_1_gray_scale) or not self.is_valid_image(frame_2_gray_scale) or not self.is_valid_image(frame_3_gray_scale)
            if not ImageEmpty:
                if not is_test_tfrecord:
                    if not self.is_valid_image(label):
                        images_without_label.append(i)
                    else:    
                        images_with_label.append(i)
                else:
                    images_without_label.append(i)
        
        if is_train_tfrecord and self.fifty_percent_of_labeled:
            num_images_with_label=len(images_with_label)
            num_images_without_label=len(images_without_label)
            num_samples=min(num_images_with_label,num_images_without_label)
            
            if num_images_with_label<=num_images_without_label:
                images_without_label=random.sample(images_without_label,num_samples)
    
            if num_images_with_label>num_images_without_label:
                images_with_label=random.sample(images_with_label,num_samples)
          
        for i in images_without_label+images_with_label:

            frame_0_gray_scale= self.get_gray_scale(frame0[:, :, i])
            frame_1_gray_scale= self.get_gray_scale(frame1[:, :, i])
            frame_2_gray_scale= self.get_gray_scale(frame2[:, :, i])
            frame_3_gray_scale= self.get_gray_scale(frame3[:, :, i])

            frames=[frame_0_gray_scale,frame_1_gray_scale,frame_2_gray_scale,frame_3_gray_scale]
            frames = np.array(frames)
            frames=frames.transpose([1,2,0])

            if is_test_tfrecord:
                label=np.zeros((self.HEIGHT,self.WIDTH))
            else:
                label=datalabel[:,:,i].astype(np.uint8)
                label = np.array(label)

            example = tf.train.Example(features = tf.train.Features(feature = {
                       'image': _bytes_feature(frames.tostring()),
                       'label': _bytes_feature(label.tostring()),
                       'PatientID': _int64_feature(int(self.get_patient_id(inputFileVolume))),
                       'Slide': _int64_feature(i),
                       'height': _int64_feature(self.HEIGHT),
                       'width': _int64_feature(self.WIDTH),
                       'depth': _int64_feature(self.DEPTH),
                       }))
            

            writer.write(example.SerializeToString())


        writer.close()
        print("Generated TFRecord: %s" % outputFile)
        
        return



    def generate_tfrecords(self):
        utils.ensure_dir(self.outputDir)
        utils.ensure_dir(self.outputImagePathTR)
        utils.ensure_dir(self.outputImagePathVL)
        utils.ensure_dir(self.outputimagePathTS)

        
        num_of_images_for_train_and_val=self.PATIENT_END_TRAINING-self.PATIENT_START_TRAINING+1
        images_for_validation=int(utils.percentage(self.percent_for_validation,num_of_images_for_train_and_val))
        #Get random list of files for validation.
        validation_list = random.sample(range(self.PATIENT_START_TRAINING, self.PATIENT_END_TRAINING+1),images_for_validation)   
        
        #train and validation
        for i,j in enumerate(list(range(self.PATIENT_START_TRAINING, self.PATIENT_END_TRAINING+1))):
            InputFileVolume=os.path.join(self.imagePathTR,"BRATS_%03d.nii.gz" % (j))
            InputFileLabel=os.path.join(self.labelPathTR,"BRATS_%03d.nii.gz" % (j))
            if os.path.isfile(InputFileVolume) and os.path.isfile(InputFileLabel):
                if j in validation_list:
                    self.generate_tfrecord_from_patient(InputFileVolume,InputFileLabel,False,True,False)
                else:
                    self.generate_tfrecord_from_patient(InputFileVolume,InputFileLabel,True,False,False)
                        
        #test
        for i,j in enumerate(list(range(self.PATIENT_START_TEST, self.PATIENT_END_TEST+1))):
            InputFileVolume=os.path.join(self.imagePathTS,"BRATS_%03d.nii.gz" % (j))
            if os.path.isfile(InputFileVolume) and os.path.isfile(InputFileLabel):
                self.generate_tfrecord_from_patient(InputFileVolume,"",False,False,True)
                        

if __name__ == '__main__':

    imagePathTR = "../BrainTumourImages/Original/imagesTr"
    labelPathTR = "../BrainTumourImages/Original/labelsTr"
    imagePathTS = "../BrainTumourImages/Original/imagesTs"
    outputDir = "../BrainTumourImages/Generated"
    number_of_volumes_to_process=500
    percent_for_validation=20
    fifty_percent_of_labeled=True
    generator = GenerateTFRedord(imagePathTR,labelPathTR,imagePathTS,outputDir,number_of_volumes_to_process,percent_for_validation,fifty_percent_of_labeled)
    generator.generate_tfrecords()