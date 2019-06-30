import os
import argparse
import tensorflow as tf
from dataset import *
from tensorflow.python.keras import backend as K
import numpy as np
import sys
from utils import utils

"""
tf.losses.softmax_cross_entropy -> tf.nn.softmax_cross_entropy_with_logits_v2 -> keras.categorical_crossentropy
tf.losses.sparse_softmax_cross_entropy -> tf.nn.sparse_softmax_cross_entropy_with_logits. -> keras.sparse_categorical_crossentropy
tf.losses.sigmoid_cross_entropy -> tf.nn.sigmoid_cross_entropy_with_logits. -> keras.binary_crossentrpy
If your targets are one-hot encoded, use categorical_crossentropy.
    Examples of one-hot encodings:
        [1,0,0]
        [0,1,0]
        [0,0,1]
If your targets are integers, use sparse_categorical_crossentropy.
    Examples of integer encodings (for the sake of completion):
        1
        2
        3
If your target is one vector of values (0,1), use Sigmoid
 OneHot  BinaryLabels  Layer Input Layer Output        Loss
 SI        SI            2            2            softmax_cross_entropy
 SI        NO            4            4            softmax_cross_entropy
 NO        SI            1            1            sigmoid_cross_entropy
 NO        NO            1            4            sparse_softmax_cross_entropy
"""

def main(trainingdir, model, num_epochs, size_batch_train, size_batch_test, size_batch_valid, logdir,perform_one_hot,binarize_labels):

    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)

    train_list, valid_list, test_list = get_file_lists(trainingdir)

    label_input_size,label_output_size=get_tensor_size(perform_one_hot,binarize_labels)

    train_dataset = create_dataset(filenames=train_list,mode="training", num_epochs=1, batch_size=size_batch_train,perform_one_hot=perform_one_hot,binarize_labels=binarize_labels)
    train_iterator = train_dataset.make_initializable_iterator()
    validation_dataset = create_dataset(filenames=valid_list,mode="validation", num_epochs=1, batch_size=size_batch_valid,perform_one_hot=perform_one_hot,binarize_labels=binarize_labels)
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_dataset = create_dataset(filenames=test_list,mode="testing", num_epochs=1, batch_size=size_batch_test,perform_one_hot=perform_one_hot,binarize_labels=binarize_labels)
    test_iterator = test_dataset.make_initializable_iterator()

    # Feedable iterator assigns each iterator a unique string handle it is going to work on
    handle = tf.placeholder(tf.string, shape = [])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    x, y = iterator.get_next()

    x.set_shape([None, 192, 192, 4])
    x = tf.cast(x, tf.float32)

    training_placeholder = tf.placeholder(dtype=tf.bool, shape=[], name='training_placeholder')

    if model == "unet_keras":
        from models import unet_keras as model
        logits, logits_soft = model.unet(x,training_placeholder,label_output_size)
    elif model == "unet_tensorflow":
        from models import unet_tensorflow as model
        logits, logits_soft = model.unet(x, training=training_placeholder, norm_option=False,drop_val=0.5,label_output_size=label_output_size)


    y.set_shape([None, 192, 192, label_input_size])
    y = tf.cast(y, tf.int32)


    # if label_input_size>1: #OneHotEncoding
        # loss_op= tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits))
        # tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
        # loss_op = tf.losses.get_total_loss(name='loss_op')
    # else: #labelEncoding
        # if label_output_size==1:
            # loss_op=tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits))
        # else:
            # loss_op= tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits))


    # if label_input_size>1: #OneHotEncoding
        # Define the IoU metrics and update operations
        # IoU_metrics, IoU_metrics_update = tf.metrics.mean_iou(labels=y, predictions=logits_soft, num_classes=label_input_size, name='my_metric_IoU')
        # Isolate the variables stored behind the scenes by the metric operation
        # running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric_IoU")
        # Define initializer to initialize/reset running variables
        # running_vars_initializer = tf.variables_initializer(var_list=running_vars)


    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train_op = tf.group([update_ops, optimizer.minimize(loss_op,global_step=global_step)])



    ######################################## SUMMARIES #########################################################

    tf.summary.image('input_0',tf.expand_dims(x[:,:,:,0],axis=-1))


    # if label_input_size==1:
        # tf.summary.image("labels",tf.cast(y,tf.float32))
    # elif label_input_size>1:
        # tf.summary.image('labels_0',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,0],axis=-1))
        # tf.summary.image('labels_1',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,1],axis=-1))
        # if label_input_size>2:
            # tf.summary.image('labels_2',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,2],axis=-1))
            # tf.summary.image('labels_3',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,3],axis=-1))


    # if label_output_size==1:
        # tf.summary.image('prediction',tf.expand_dims(logits_soft[:,:,:,0],axis=-1))
    # elif label_output_size>1:
        # tf.summary.image("prediction", logits_soft[:,:,:,1:])
        # tf.summary.image('prediction_0',tf.expand_dims(logits_soft[:,:,:,0],axis=-1))
        # tf.summary.image('prediction_1',tf.expand_dims(logits_soft[:,:,:,1],axis=-1))
        # if label_output_size>2:
            # tf.summary.image('prediction_2',tf.expand_dims(logits_soft[:,:,:,2],axis=-1))
    tf.summary.image('logits',tf.expand_dims(logits_soft[:,:,:,0],axis=-1))

    #tf.summary.histogram("logits",logits)

    # tf.summary.scalar("loss", loss_op)

    summary_op=tf.summary.merge_all()

    # op to write logs to Tensorboard
    logdir = os.path.expanduser(logdir)
    utils.ensure_dir(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
    # writer_val =tf.summary.FileWriter(os.path.join(logdir, 'validation loss'), graph=tf.get_default_graph())


    ######################################## RUN SESSION #########################################################

    with tf.Session() as sess:

        # Initialize Variables
        # if restore_weights:
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
        # else:
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        #training, validation and saving
        # for epoch in range(num_epochs):
            # sess.run(train_iterator.initializer)
            # step=0
            try:
                sess.run(logits_soft,feed_dict={x:x_test_batch})
            
                # while True:

                    # train
                    # _,cost,summary_val,step_gl,logits_val,_ = sess.run([train_op,loss_op,summary_op,global_step,logits,logits_soft], feed_dict={handle: train_handle,training_placeholder: True})

                    # writer.add_summary(summary_val,step_gl)

                    # step += 1
                    # print('\n Training step: Epoch {}, batch {} -- Loss: {:.3f}'.format(epoch+1, step, cost))



                    # validation
                    # if step % step_metrics == 0:
                        # total_validation_loss = [] #list where we will store the loss at each batch
                        # sess.run(validation_iterator.initializer)
                        # step_val=0
                        # print('\n Step {}: Saving weights to {}'.format(step, model_checkpoint_path))
                        # initialize/reset the running variables of the IoU metrics
                        # sess.run(running_vars_initializer)
                        # try:
                            # while True:
                                # cost_valid, _ = sess.run([loss_op, IoU_metrics_update], feed_dict={handle: validation_handle,training_placeholder: False})
                                # total_validation_loss.append(cost_valid)
                                # step_val += 1
                                # print('\nValidation step: Epoch {}, batch {} -- Loss: {:.3f}'.format(epoch+1, step_val, cost_valid))
                        # except tf.errors.OutOfRangeError:
                            # pass
                        # loss
                        # total_validation_loss = np.mean(total_validation_loss)
                        # validation_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=total_validation_loss)])
                        # writer_val.add_summary(validation_loss_summary,step_gl)

                        # IoU metrics
                        # if label_input_size>1: #OneHotEncoding
                            # IoU metrics
                            # IoU_score = sess.run(IoU_metrics)
                            # IoU_summary = tf.Summary(value=[tf.Summary.Value(tag="IoU_metrics", simple_value=IoU_score)])
                            # writer.add_summary(IoU_summary,step_gl)
                            # print('\n Epoch {} and training batch {} -- Validation loss {:.3f} and IoU metrics {:.3f}'.format(epoch+1, step,total_validation_loss, IoU_score))
                        # else:
                            # print('\n Epoch {} and training batch {} -- Validation loss {:.3f}'.format(epoch+1, step,total_validation_loss))

                    # saving
                    # if step % steps_saver == 0:
                        # print('\n Step {}\tSaving weights to {}'.format(step+1, model_checkpoint_path))
                        # saver.save(sess, save_path=model_checkpoint_path,global_step=global_step)

            except tf.errors.OutOfRangeError:
                pass

    return