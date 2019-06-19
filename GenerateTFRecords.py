import nibabel as nib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import utils
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

    :param number_of_images_to_process: The number of images to process. both for training, for testing.

    :param percent_for_validation: Percentaje of the resulting images that will be reserved for validation.
                                   Ej. If value is 5 (5%) and a single patient has 100 slices, then 5 slices will be reserved and saved in a TFRecord file for the patient in the validation path.
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

    def __init__(self, image_path_tr,label_path_tr,image_path_ts,output_dir,number_of_images_to_process,percent_for_validation=5):
        self.imagePathTR = image_path_tr
        self.labelPathTR = label_path_tr
        self.imagePathTS=image_path_ts
        self.outputDir=output_dir
        self.outputImagePathTR=os.path.join(self.outputDir,"Training")
        self.outputImagePathVL=os.path.join(self.outputDir,"Validation")
        self.outputimagePathTS=os.path.join(self.outputDir,"Test")
        self.number_of_images_to_process=number_of_images_to_process
        self.percent_for_validation=percent_for_validation

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


    def generate_tfrecord_from_patient(self,inputFileVolume,InputFileLabel,is_test_tfrecord):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        outputFile=os.path.basename(inputFileVolume).replace("nii.gz", "tfrecords")
        outputFileValidation=os.path.join(self.outputImagePathVL,outputFile)

        if is_test_tfrecord:
            outputFile=os.path.join(self.outputimagePathTS,outputFile)
        else:
            outputFile=os.path.join(self.outputImagePathTR,outputFile)

        writer = tf.python_io.TFRecordWriter(outputFile)
        if not is_test_tfrecord:
            writer_validation = tf.python_io.TFRecordWriter(outputFileValidation)

        nii_vol = nib.load(inputFileVolume)
        dataVol = nii_vol.get_data()

        if not is_test_tfrecord:
            nii_label = nib.load(InputFileLabel)
            datalabel = nii_label.get_data()

        frame0=dataVol[:,:,:,0]
        frame1=dataVol[:,:,:,1]
        frame2=dataVol[:,:,:,2]
        frame3=dataVol[:,:,:,3]

        valid_images = []
        for i in range(self.SLICE_START, self.SLICE_END+1):

            frame_0_gray_scale= self.get_gray_scale(frame0[:, :, i])
            frame_1_gray_scale= self.get_gray_scale(frame1[:, :, i])
            frame_2_gray_scale= self.get_gray_scale(frame2[:, :, i])
            frame_3_gray_scale= self.get_gray_scale(frame3[:, :, i])
            ImageEmpty=not self.is_valid_image(frame_0_gray_scale) or not self.is_valid_image(frame_1_gray_scale) or not self.is_valid_image(frame_2_gray_scale) or not self.is_valid_image(frame_3_gray_scale)
            if not ImageEmpty:
                valid_images.append(i)


        if not is_test_tfrecord:
            number_of_images_for_validation=int(utils.percentage(self.percent_for_validation,len(valid_images)))
            validation_list = random.sample(valid_images,number_of_images_for_validation)
        else:
            validation_list =[]

        for i in valid_images:

            frame_0_gray_scale= self.get_gray_scale(frame0[:, :, i])
            frame_1_gray_scale= self.get_gray_scale(frame1[:, :, i])
            frame_2_gray_scale= self.get_gray_scale(frame2[:, :, i])
            frame_3_gray_scale= self.get_gray_scale(frame3[:, :, i])

            frames=[frame_0_gray_scale,frame_1_gray_scale,frame_2_gray_scale,frame_3_gray_scale]
            frames = np.array(frames)
            frames=frames.transpose([1,2,0])

            if not is_test_tfrecord:
                label=datalabel[:,:,i].astype(np.uint8)
                label = np.array(label)
            else:
                label=np.zeros((self.HEIGHT,self.WIDTH))

            example = tf.train.Example(features = tf.train.Features(feature = {
                       'image': _bytes_feature(frames.tostring()),
                       'label': _bytes_feature(label.tostring()),
                       'PatientID': _int64_feature(int(self.get_patient_id(inputFileVolume))),
                       'Slide': _int64_feature(i),
                       'height': _int64_feature(self.HEIGHT),
                       'width': _int64_feature(self.WIDTH),
                       'depth': _int64_feature(self.DEPTH),
                       }))

            if not i in validation_list:
                writer.write(example.SerializeToString())
            else:
                writer_validation.write(example.SerializeToString())

        writer.close()
        print("Generated TFRecord: %s" % outputFile)

        if not is_test_tfrecord:
            writer_validation.close()
            print("Generated TFRecord: %s" % outputFileValidation)


        return

    def generate_tfrecords_for_training(self):
        for i,j in enumerate(list(range(self.PATIENT_START_TRAINING, self.PATIENT_END_TRAINING+1))):
            if i<self.number_of_images_to_process:
                inputFileVolume=os.path.join(self.imagePathTR,"BRATS_%03d.nii.gz" % (j))
                InputFileLabel=os.path.join(self.labelPathTR,"BRATS_%03d.nii.gz" % (j))
                if os.path.isfile(inputFileVolume) and os.path.isfile(InputFileLabel):
                    self.generate_tfrecord_from_patient(inputFileVolume,InputFileLabel,False)


    def generate_tfrecords_for_test(self):
        for i,j in enumerate(list(range(self.PATIENT_START_TEST, self.PATIENT_END_TEST+1))):
            if i<self.number_of_images_to_process:
                inputFileVolume=os.path.join(self.imagePathTS,"BRATS_%03d.nii.gz" % (j))
                if os.path.isfile(inputFileVolume):
                    self.generate_tfrecord_from_patient(inputFileVolume,"",True)

    def generate_tfrecords(self):
        utils.ensure_dir(self.outputDir)
        utils.ensure_dir(self.outputImagePathTR)
        utils.ensure_dir(self.outputImagePathVL)
        utils.ensure_dir(self.outputimagePathTS)
        self.generate_tfrecords_for_training()
        self.generate_tfrecords_for_test()


if __name__ == '__main__':

    imagePathTR = "Macintosh_SSD_Samsung_EVO_256_GB/BrainTumourImages/Original/imagesTr"
    labelPathTR = "Macintosh_SSD_Samsung_EVO_256_GB/BrainTumourImages/Original/labelsTr"
    imagePathTS = "Macintosh_SSD_Samsung_EVO_256_GB/BrainTumourImages/Original/imagesTs"
    outputDir = "Macintosh_SSD_Samsung_EVO_256_GB/BrainTumourImages/Generated"
    number_of_images_to_process=485
    percent_for_validation=5
    generator = GenerateTFRedord(imagePathTR,labelPathTR,imagePathTS,outputDir,number_of_images_to_process,percent_for_validation)
    generator.generate_tfrecords()
