#https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
import tensorflow as tf

#Data pipelin using TFRecords

def input_fn(filenames, perform_shuffle=False, num_epochs=1, batch_size=1):

    def _parse_function(serialized):
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

        sample = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        frames = tf.decode_raw(sample['image'], tf.uint8)
        label = tf.decode_raw(sample['label'], tf.uint8)
        PatientID = tf.cast(sample['PatientID'], tf.int32)
        Slide = tf.cast(sample['Slide'], tf.int32)
        height= tf.cast(sample['height'], tf.int32)
        width= tf.cast(sample['width'], tf.int32)
        depth= tf.cast(sample['depth'], tf.int32)

        frames = tf.reshape(frames,(height, width, depth))
        label = tf.reshape(label,(height, width,-1))
        tf.expand_dims(label, 1).shape

        #Add data augmentation here
        frames=tf.image.central_crop(frames,0.8)
        label=tf.image.central_crop(label,0.8)

        return frames, label


    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:

        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    # Repeats dataset this # times
    dataset = dataset.repeat(num_epochs)

    # Batch size to use
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
