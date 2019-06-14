import tensorflow as tf
import os
import shutil
from PIL import Image
import numpy as np
from GenerarImatges import outputDir

class TFRecordExtractor:

    """Class for extract images from a TFRecord file.
       The structure of the TFRecord must be:
                       'image': _bytes_feature(frames.tostring()),
                       'label': _bytes_feature(label.tostring()),
                       'PatientID': _int64_feature(int(self.get_patient_id(inputFileVolume))),
                       'Slide': _int64_feature(i),
                       'height': _int64_feature(self.HEIGHT),
                       'width': _int64_feature(self.WIDTH),
                       'depth': _int64_feature(self.DEPTH),

    :param tfrecord_file: TFRecord file to extract.

    :param isTraining:(True/False) If True, then the label will be extracted. It's also used to generate the name of the generated files.

    :param withTFSession:(True/False) Two options can be used. With or without using a TF session.
    """

    def __init__(self, tfrecord_file,outputFolder,isTraining,withTFSession=True):
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.outputFolder = outputFolder
        self.isTraining=isTraining
        self.withTFSession=withTFSession

    def save_image(self, data, outputFile):
        im=Image.fromarray(data)
        im.save(outputFile)

    def get_output_file_name(self,inputFile, index, frame, origin):
        str_fileName = inputFile.split('.')[0]
        str_outputFile = '%s-%s%03d-slice%03d.%s' % (
                                                    str_fileName,
                                                    origin,
                                                    frame, index,
                                                    "jpg")

        return str_outputFile

    def export_without_session(self):
        record_iterator = tf.python_io.tf_record_iterator(path=self.tfrecord_file)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            frames = (example.features.feature['image']
                                    .bytes_list
                                    .value[0])
            label = (example.features.feature['label']
                                    .bytes_list
                                    .value[0])
            PatientID = (example.features.feature['PatientID']
                                    .int64_list
                                    .value[0])
            Slide = (example.features.feature['Slide']
                                    .int64_list
                                    .value[0])
            width = (example.features.feature['width']
                                    .int64_list
                                    .value[0])
            height = (example.features.feature['height']
                                    .int64_list
                                    .value[0])
            depth = (example.features.feature['depth']
                                    .int64_list
                                    .value[0])


            frames = np.fromstring(frames, dtype=np.uint8)
            frames = frames.reshape(height,width, depth)
            if self.isTraining:
                label = np.fromstring(label, dtype=np.uint8)
                label = label.reshape(height,width)

            PatientID=int(PatientID)
            Slide =int(Slide)



            outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,0,'train' if self.isTraining else 'test')
            self.save_image(frames[:, :,0],os.path.join(self.outputFolder,outputFile))
            print("Generated file: %s" % outputFile)

            outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,1,'train' if self.isTraining else 'test')
            self.save_image(frames[:, :,1],os.path.join(self.outputFolder,outputFile))
            print("Generated file: %s" % outputFile)

            outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,2,'train' if self.isTraining else 'test')
            self.save_image(frames[:, :,2],os.path.join(self.outputFolder,outputFile))
            print("Generated file: %s" % outputFile)

            outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,3,'train' if self.isTraining else 'test')
            self.save_image(frames[:, :,3],os.path.join(self.outputFolder,outputFile))
            print("Generated file: %s" % outputFile)

            if self.isTraining:
                outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,0,"label")
                self.save_image(label,os.path.join(self.outputFolder,outputFile))
                print("Generated file: %s" % outputFile)


    def export_with_session(self):
        def extract_fn(tfrecord):
            # Extract features using the keys set during creation
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'PatientID': tf.FixedLenFeature([], tf.int64),
                'Slide': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            }

            # Extract the data record
            sample = tf.parse_single_example(tfrecord, features)
            frames = tf.decode_raw(sample['image'], tf.uint8)
            label = tf.decode_raw(sample['label'], tf.uint8)
            PatientID = tf.cast(sample['PatientID'], tf.int32)
            Slide = tf.cast(sample['Slide'], tf.int32)
            height = tf.cast(sample['height'], tf.int32)
            width = tf.cast(sample['width'], tf.int32)
            depth = tf.cast(sample['depth'], tf.int32)

            return [frames, label, PatientID, Slide, height, width, depth]

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(extract_fn)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #try:
                # Keep extracting data till TFRecord is exhausted
            while True:
                try:
                    image_data = sess.run(next_image_data)
                    frames = np.fromstring(image_data[0], dtype=np.uint8)
                    label = np.fromstring(image_data[1], dtype=np.uint8)
                    Slide=int(image_data[3])
                    height=int(image_data[4])
                    width=int(image_data[5])
                    depth=int(image_data[6])

                    frames= frames.reshape(height, width, 4)
                    if self.isTraining:
                        label = label.reshape(height, width)



                    outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,0,'train' if self.isTraining else 'test')
                    self.save_image(frames[:, :, 0],os.path.join(self.outputFolder,outputFile))
                    print("Generated file: %s" % outputFile)

                    outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,1,'train' if self.isTraining else 'test')
                    self.save_image(frames[:, :, 1],os.path.join(self.outputFolder,outputFile))
                    print("Generated file: %s" % outputFile)

                    outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,2,'train' if self.isTraining else 'test')
                    self.save_image(frames[:, :, 2],os.path.join(self.outputFolder,outputFile))
                    print("Generated file: %s" % outputFile)

                    outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,3,'train' if self.isTraining else 'test')
                    self.save_image(frames[:, :, 3],os.path.join(self.outputFolder,outputFile))
                    print("Generated file: %s" % outputFile)

                    if self.isTraining:
                        outputFile=self.get_output_file_name(os.path.basename(self.tfrecord_file),Slide,0,"label")
                        self.save_image(label,os.path.join(self.outputFolder,outputFile))
                        print("Generated file: %s" % outputFile)
                except tf.errors.OutOfRangeError:
                    break
            #except:
            #    pass

    def extract_image(self):
        if self.withTFSession==True:
            self.export_with_session()
        else:
            self.export_without_session()


if __name__ == '__main__':
    tfrecord_file='/Users/aitorjara/Desktop/CleanSlices/TFRecords/'
    outputFolder='/Users/aitorjara/Desktop/CleanSlices/ExtractTFRecords/'
    t = TFRecordExtractor(tfrecord_file,outputFolder,True)
    t.extract_image()
