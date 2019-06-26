import argparse
import os
import numpy as np
import tensorflow as tf
import utils
import Generate_Batch


BATCH_SIZE_TRAIN = 25
BATCH_SIZE_TEST = 20
NUM_EPOCHS = 1
STEPS_LOSS_LOG = 50
STEPS_SAVER = 100
LEARNING_RATE= 1e-4
TRAININGDIR= '/Users/aitorjara/Desktop//BrainTumourImages/Generated/'
LOGDIR='/tmp/aidl'


   
# Input pipeline
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
    
# Model
def unet_model(data, training=False, norm_option=False, drop_val=0.5):    
    if norm_option != True and norm_option != False:
        sys.exit('Not a valid norm_option')
    if drop_val<0 or drop_val>1:
        sys.exit('Not a valid drop_val')    
    #Downsampling path (Convolution layers)
    conv1 = tf.layers.conv2d(data, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal())
    if norm_option == True:
        conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(conv1, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
    if norm_option == True:
        conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)
    pool1 = tf.layers.max_pooling2d(conv2, [2,2], strides=[2,2], padding='SAME')    
    conv3 = tf.layers.conv2d(pool1, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
    if norm_option == True:
        conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.relu(conv3)
    conv4 = tf.layers.conv2d(conv3, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        conv4 = tf.layers.batch_normalization(conv4)
    conv4 = tf.nn.relu(conv4)
    pool2 = tf.layers.max_pooling2d(conv4, [2,2], strides=[2,2], padding='SAME')    
    conv5 = tf.layers.conv2d(pool2, 256, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        conv5 = tf.layers.batch_normalization(conv5)
    conv5 = tf.nn.relu(conv5)
    conv6 = tf.layers.conv2d(conv5, 256, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        conv6 = tf.layers.batch_normalization(conv6)
    conv6 = tf.nn.relu(conv6)
    pool3 = tf.layers.max_pooling2d(conv6, [2,2], strides=[2,2], padding='SAME')    
    conv7 = tf.layers.conv2d(pool3, 512, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        conv7 = tf.layers.batch_normalization(conv7)
    conv7 = tf.nn.relu(conv7)
    conv8 = tf.layers.conv2d(conv7, 512, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*512))))
    if norm_option == True:
        conv8 = tf.layers.batch_normalization(conv8)
    conv8 = tf.nn.relu(conv8)
    pool4 = tf.layers.max_pooling2d(conv8, [2,2], strides=[2,2], padding='SAME')    
    #Bottleneck
    conv_bottle_1 = tf.layers.conv2d(pool4, 1024, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*512))))
    if norm_option == True:
        conv_bottle_1 = tf.layers.batch_normalization(conv_bottle_1)
    conv_bottle_1 = tf.nn.relu(conv_bottle_1)
    conv_bottle_2 = tf.layers.conv2d(conv_bottle_1, 1024, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*1024))))
    if norm_option == True:
        conv_bottle_2 = tf.layers.batch_normalization(conv_bottle_2)
    conv_bottle_2 = tf.nn.relu(conv_bottle_2)
    if training == True:
        drop = tf.layers.dropout(conv_bottle_2, rate=drop_val)
    if training == False:
        drop = conv_bottle_2    
    #Upsampling path (Deconvolutional layers)
    deconv1 = tf.layers.conv2d_transpose(drop, 1024, [2,2], strides=[2,2], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*1024))))
    if norm_option == True:
        deconv1 = tf.layers.batch_normalization(deconv1)
    deconv1 = tf.nn.relu(deconv1)
    concat1 = tf.stack([conv8,deconv1], axis=-1)
    UpPath_conv1 = tf.layers.conv2d(concat1, 512, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*1024))))
    if norm_option == True:
        UpPath_conv1 = tf.layers.batch_normalization(UpPath_conv1)
    UpPath_conv1 = tf.nn.relu(UpPath_conv1)
    UpPath_conv2 = tf.layers.conv2d(UpPath_conv1, 512, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*512))))
    if norm_option == True:
        UpPath_conv2 = tf.layers.batch_normalization(UpPath_conv2)
    UpPath_conv2 = tf.nn.relu(UpPath_conv2)    
    deconv2 = tf.layers.conv2d_transpose(UpPath_conv2, 512, [2,2], strides=[2,2], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*512))))
    if norm_option == True:
        deconv2 = tf.layers.batch_normalization(deconv2)
    deconv2 = tf.nn.relu(deconv2)
    concat2 = tf.stack([conv6,deconv2], axis=-1)
    UpPath_conv3 = tf.layers.conv2d(concat2, 256, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*512))))
    if norm_option == True:
        UpPath_conv3 = tf.layers.batch_normalization(UpPath_conv3)
    UpPath_conv3 = tf.nn.relu(UpPath_conv3)
    UpPath_conv4 = tf.layers.conv2d(UpPath_conv3, 256, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        UpPath_conv4 = tf.layers.batch_normalization(UpPath_conv4)
    UpPath_conv4 = tf.nn.relu(UpPath_conv4)    
    deconv3 = tf.layers.conv2d_transpose(UpPath_conv4, 256, [2,2], strides=[2,2], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        deconv3 = tf.layers.batch_normalization(deconv3)
    deconv3 = tf.nn.relu(deconv3)
    concat3 = tf.stack([conv4,deconv3], axis=-1)
    UpPath_conv5 = tf.layers.conv2d(concat3, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        UpPath_conv5 = tf.layers.batch_normalization(UpPath_conv5)
    UpPath_conv5 = tf.nn.relu(UpPath_conv5)
    UpPath_conv6 = tf.layers.conv2d(UpPath_conv5, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        UpPath_conv6 = tf.layers.batch_normalization(UpPath_conv6)
    UpPath_conv6 = tf.nn.relu(UpPath_conv6)    
    deconv4 = tf.layers.conv2d_transpose(UpPath_conv6, 128, [2,2], strides=[2,2], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        deconv4 = tf.layers.batch_normalization(deconv4)
    deconv4 = tf.nn.relu(deconv4)
    concat4 = tf.stack([conv2,deconv4], axis=-1)
    UpPath_conv7 = tf.layers.conv2d(concat4, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        UpPath_conv7 = tf.layers.batch_normalization(UpPath_conv7)
    UpPath_conv7 = tf.nn.relu(UpPath_conv7)
    UpPath_conv8 = tf.layers.conv2d(UpPath_conv7, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
    if norm_option == True:
        UpPath_conv8 = tf.layers.batch_normalization(UpPath_conv8)
    UpPath_conv8 = tf.nn.relu(UpPath_conv8)
    UpPath_conv9 = tf.layers.conv2d(UpPath_conv8, 4, [1,1], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
    if norm_option == True:
        UpPath_conv9 = tf.layers.batch_normalization(UpPath_conv9)
    UpPath_conv9 = tf.nn.relu(UpPath_conv9)    
    return UpPath_conv9   

#NºExamples
def count_records(path):
    
    #function that returns the number of records in a set of TFRecords stored a directory. We will use it to count the number of
    #training and validation data 
    
    num = 0
    for record_file in os.listdir(path):
        TFRecord_path = os.path.join(path,record_file)
        for record in tf.io.tf_record_iterator(TFRecord_path):
            num += 1
    return num

# Metric
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
    
def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    dice=0
    for index in range(numLabels):
        dice -= dice_coeff(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])
    return dice  

# Loss
def loss_rara1(labels, logits):
    #https://github.com/perslev/MultiPlanarUNet/
    # Flatten
    labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
    logits = tf.reshape(logits, [-1, 4])    
    # Calculate in-batch class counts and total counts
    target_one_hot = tf.one_hot(labels, 4)
    counts = tf.cast(tf.reduce_sum(target_one_hot, axis=0), tf.float32)
    total_counts = tf.reduce_sum(counts)    
    # Compute balanced sample weights
    weights = (tf.ones_like(counts) * total_counts) / (counts * 4)    
    # Compute sample weights from class weights
    weights = tf.gather(weights, labels)    
    #return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)    
     
def loss_rara2(labels, logits):
    #https://github.com/tensorflow/tensorflow/issues/10021
    #https://stackoverflow.com/questions/40198364/how-can-i-implement-a-weighted-cross-entropy-loss-in-tensorflow-using-sparse-sof/46984951#46984951
    class_weights = tf.constant([0.1 , 0.3 , 0.3 , 0.3])  # 3 classes
    sample_weights = tf.gather(class_weights, labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits,weights=sample_weights)  

def loss_sparse(labels, logits):
    labels = backend.print_tensor(labels, message='labels = ')
    logits = backend.print_tensor(logits, message='logits = ')  
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) 
    return loss
    #return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
 

def main(trainingdir, num_epochs, batch_size, learning_rate, logdir, restore_weights): 
    ''':param step_valid:   after how many batches of training images we perform the validation
       :param step_metrics: after how many batches of training images we keep track of the summary'''

     # ----------------- TRAINING LOOP SETUP ---------------- #
    logdir = os.path.expanduser(logdir)
    utils.ensure_dir(logdir)
    
    # ------------------ DEFINITION PHASE ------------------ #
    global_step  = tf.get_variable('global_step', dtype=tf.int32, initializer=0, trainable=False)
    
    #Images
    train_images = count_records(os.path.join(trainingdir, 'Training')) #number of training images  
    valid_images = count_records(os.path.join(trainingdir, 'Validation')) #number of validation images
    size_batch_valid = int(valid_images/(int(train_images/size_batch_train)/step_valid)) #param step_valid means that every step_valid batches of training images, a batch of image validation is going to be perfomed
    tf.summary.image("input_0",tf.expand_dims(x[:,:,:,0],axis=-1))
    tf.summary.image("labels",tf.cast(y,tf.float32))
    tf.summary.image("prediction", logits[:,:,:,1:])
    tf.summary.histogram("logits",logits)
    
    # Loss
    loss = tf.reduce_mean(loss_sparse(labels=y, logits=logits)) #loss = tf.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss, global_step=global_step )
    
    # Summary writer
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    summary_op = tf.summary.merge_all()
    
    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'BrainScan')
    saver = tf.train.Saver()
    
    
    # ----------------- RUN PHASE ------------------- #
    with tf.Session() as sess:
        if restore_weights:
            saver.restore(sess, restore_weights)
        else:
            sess.run(tf.global_variables_initializer())
    try:
        while True:
            # Run the train step
            _, loss, step, summ_val = sess.run([train_step, loss, global_step, summary_op])
            # Print how the loss is evolving per step in order to check if the model is converging
            if step % STEPS_LOSS_LOG == 0:
                print('Step {}\tLoss {}'.format(step, loss))
                writer.add_summary(summ_val, loss,  global_step=step)
            # Save the graph definition and its weights
            if step % STEPS_SAVER == 0:
                print('Step {}\tSaving weights to{}'.format(step, model_checkpoint_path))
                saver.save(sess, save_path=model_checkpoint_path, global_step=global_step)
    except tf.errors.OutOfRangeError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('trainingdir', default=TRAININGDIR, help='Path to the CSV decribing the dataset')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-b', '--batch_size_train', type=int, default=BATCH_SIZE_TRAIN, help='Batch size for training')
    parser.add_argument('-b', '--batch_size_test', type=int, default=BATCH_SIZE_TEST, help='Batch size for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-r', '--restore', help='Path to model checkpoint to restore weights from.')
   
    args = parser.parse_args()

    main(args.trainingdir, args.num_epochs, args.batch_size, args.learning_rate, args.logdir, args.restore)