#https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
#https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
#http://androidkt.com/train-keras-model-with-tensorflow-estimators-and-datasets-api/
#https://github.com/Tony607/Keras_catVSdog_tf_estimator/blob/master/keras_estimator_vgg16-cat_vs_dog.ipynb
#http://androidkt.com/feeding-your-own-data-set-into-the-cnn-model-in-tensorflow/


#data augmentation
#https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb#scrollTo=tkNqQaR2HQbd
#http://androidkt.com/tensorflow-image-augmentation-using-tf-image/

import tensorflow as tf
import model as model
import os
import utils
from tensorflow.keras import models
from tensorflow.keras import layers
from PIL import Image
#from tensorflow.keras.preprocessing import image
import dataset


NUM_EPOCHS =10
BATCH_SIZE=10


def get_file_lists(data_dir):
    import glob
    train_list = glob.glob(data_dir + '/Training/' + '*')
    valid_list = glob.glob(data_dir + '/Validation/' + '*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list

def train_input_fn(file_path):
    return dataset.input_fn(file_path,True,NUM_EPOCHS,BATCH_SIZE)

def validation_input_fn(file_path):
    return dataset.input_fn(file_path,False,NUM_EPOCHS,BATCH_SIZE)

