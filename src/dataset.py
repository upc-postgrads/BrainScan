#Data pipeline using TFRecords

import tensorflow as tf
import glob


def get_tensor_size(perform_one_hot,binarize_labels):
    
    if perform_one_hot:    
        if binarize_labels:
            return 2,2
        else:
            return 4,4
    else:
        if binarize_labels:
            return 1,1
        else:
            return 1,4   
        

def get_file_lists(data_dir):    

    train_list = glob.glob(data_dir + '/Training/' + '*')
    valid_list = glob.glob(data_dir + '/Validation/' + '*')
    test_list = glob.glob(data_dir + '/Test/' + '*')
    if len(train_list) == 0 and len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list, test_list




def create_dataset(filenames, mode, num_epochs=1, batch_size=1,perform_shuffle=False,perform_one_hot=True,binarize_labels=False):
    if mode == 'validation':
        perform_shuffle = False
    if mode == 'training':
        perform_shuffle = True
    if mode == 'testing':
        perform_shuffle = False


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

        sample = tf.parse_single_example(serialized=serialized, features=features)
        frames = tf.decode_raw(sample['image'], tf.uint8)
        label = tf.decode_raw(sample['label'], tf.uint8)
        PatientID = tf.cast(sample['PatientID'], tf.int32)
        Slide = tf.cast(sample['Slide'], tf.int32)
        height= tf.cast(sample['height'], tf.int32)
        width= tf.cast(sample['width'], tf.int32)
        depth= tf.cast(sample['depth'], tf.int32)

        frames = tf.reshape(frames,(height, width, depth))
        label = tf.reshape(label,(height, width,-1))

        
        # Binarize label to only have value 0 or 1 (label > 0)
        binary_labels = tf.cast(label > 0, dtype=tf.int32)
       
        #Add data augmentation here
        frames=tf.image.central_crop(frames,0.8)
        label=tf.image.central_crop(label,0.8)
        if binarize_labels:
            binary_labels=tf.image.central_crop(binary_labels,0.8)
        
        if perform_one_hot:
            label=tf.one_hot(indices=tf.squeeze(label), depth=4)
            if binarize_labels:
                binary_labels=tf.one_hot(indices=tf.squeeze(binary_labels), depth=2)
        
        if binarize_labels:
            return frames, binary_labels
        else:
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

    return dataset
