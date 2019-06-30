import os
import argparse
import tensorflow as tf
from dataset import *
from tensorflow.python.keras import backend as K
import numpy as np
import sys
from utils import utils


def main(trainingdir, model, num_epochs, size_batch_test, logdir, logdir_w, perform_one_hot,binarize_labels):

    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)

    train_list, valid_list, test_list = get_file_lists(trainingdir)

    label_input_size,label_output_size=get_tensor_size(perform_one_hot,binarize_labels)

  
    test_dataset = create_dataset(filenames=test_list,mode="testing", num_epochs=1, batch_size=size_batch_test,perform_one_hot=perform_one_hot,binarize_labels=binarize_labels)
    test_iterator = test_dataset.make_initializable_iterator()

    # Feedable iterator assigns each iterator a unique string handle it is going to work on
    handle = tf.placeholder(tf.string, shape = [])
    iterator = tf.data.Iterator.from_string_handle(handle, test_dataset.output_types, test_dataset.output_shapes)
    x, _ = iterator.get_next()

    x.set_shape([None, 192, 192, 4])
    x = tf.cast(x, tf.float32)

    training_placeholder = tf.placeholder(dtype=tf.bool, shape=[], name='training_placeholder')

    if model == "unet_keras":
        from models import unet_keras as model
        logits, logits_soft = model.unet(x,training_placeholder,label_output_size)
    elif model == "unet_tensorflow":
        from models import unet_tensorflow as model
        logits, logits_soft = model.unet(x, training=training_placeholder, norm_option=False,drop_val=0.5,label_output_size=label_output_size)



    ######################################## SUMMARIES #########################################################

    tf.summary.image('input_0',tf.expand_dims(x[:,:,:,0],axis=-1))
    tf.summary.image('logits',tf.expand_dims(logits_soft[:,:,:,0],axis=-1))


    summary_test=tf.summary.merge_all()

    # op to write logs to Tensorboard
    logdir_w = os.path.expanduser(logdir_w)
    utils.ensure_dir(logdir_w)
    writer = tf.summary.FileWriter(logdir_w, graph=tf.get_default_graph())

    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'Checkpoint')
    saver = tf.train.Saver()

    ######################################## RUN SESSION #########################################################

    with tf.Session() as sess:

        # Initialize Variables
        #restore_weights:
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
        # else:
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())


        test_handle = sess.run(test_iterator.string_handle())


        sess.run(test_iterator.initializer)
 
        try:
            print("there")
            while True:
                summary_val,logits_test = sess.run([summary_test,logits_soft],feed_dict={handle:test_handle,training_placeholder:False})
                      
                writer.add_summary(summary_val)

        except tf.errors.OutOfRangeError:
            print("here")
   
    return