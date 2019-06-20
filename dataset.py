#https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
import tensorflow as tf
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import imageio

#Data pipelin using TFRecords

def input_fn(filenames, perform_shuffle=False, num_epochs=1, batch_size=1):

    def _parse_function(serialized):
        with tf.name_scope('create_samples'):
            features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
                'PatientID': tf.FixedLenFeature([], tf.int64),
                'Slide': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
            }
            sample = tf.parse_single_example(serialized=serialized,features=features)

            frames = tf.decode_raw(sample['image'], tf.uint8)
            label = tf.decode_raw(sample['label'], tf.uint8)
            PatientID = tf.cast(sample['PatientID'], tf.int32)
            Slide = tf.cast(sample['Slide'], tf.int32)
            height= tf.cast(sample['height'], tf.int32)
            width= tf.cast(sample['width'], tf.int32)
            depth= tf.cast(sample['depth'], tf.int32)

        with tf.name_scope('preprocessing'):
            frames = tf.reshape(frames,(height, width, depth))
            label = tf.reshape(label,(height, width,-1))
            tf.expand_dims(label, 1).shape

        #Add data augmentation here
        with tf.name_scope('data_augmentation'):
        #@tf.function()
        #def customized_cropp():
            #if mode = 'train':
                #r_path = r_path_train
            #elif mode = 'test':
                #r_path = r_path_test
            #elif mode = 'label':
                #r_path = r_path_label
            #
            #max_hei = 0
            #max_wid = 0
            #cropped_images = []
            #for filename in os.listdir(r_path):
                #data = img.imread(r_path + filename)
                #mask = data > 0             # Mask of non-black pixels (assuming image has a single channel).
                #coords = np.argwhere(mask)  # Coordinates of non-black pixels.
                #try:                        # Bounding box of non-black pixels.
                    #x0, y0 = coords.min(axis=0)
                    #x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
                #except:
                    #pass
            #
                #cropped = data[x0:x1, y0:y1]          # Get the contents of the bounding box.
                #cropped_images.append(cropped)        # Save the cropped images in a list as data
                #if cropped.shape[0] > max_hei:
                    #max_hei = cropped.shape[0]
                #if cropped.shape[1] > max_hei:
                    #max_wid = cropped.shape[1]
            #    black = [0,0,0]
                #for image in cropped_images:
                    #constant=cv2.copyMakeBorder(image, (max_hei - image.shape[0])/2, (max_wid - image.shape[0])/2, (max_wid -    image.shape[0])/2, (max_wid - #image.shape[0])/2, cv2.BORDER_CONSTANT,value=black)
            frames=tf.image.central_crop(frames,0.8)
            label=tf.image.central_crop(label,0.8)
        return frames, label


    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    if perform_shuffle:

        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    # Repeats dataset this # times
    dataset = dataset.repeat(num_epochs)

    # Batch size to use
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=10)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


#entrada labels 4 canales, softmax_cross_entropy, 50% imagenes tumor/no_tumor,
#1epoch de training y 1epoch de validation
#moure summary op.global_step, logits a cada 50
#metrica IoU
#hacer crops de las zonas de tumor
