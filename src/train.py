import os
import argparse
import tensorflow as tf
from dataset import *
from tensorflow.python.keras import backend as K
import numpy as np
import sys
from utils import utils



#########################################################

#Training parameters
NUM_EPOCHS = 5
BATCH_SIZE_TRAIN = 25
BATCH_SIZE_TEST = 2
BATCH_SIZE_VALID = 25
STEP_METRICS = 50
LEARNING_RATE = 1e-5
STEPS_SAVER = 100
MODEL_TO_USE = "unet_keras"
RESTORE_WEIGHTS=False
TRAININGDIR = "/path_of_training_images"
LOGDIR = '/tmp'
PERFORM_ONE_HOT=True
BINARIZE_LABELS=True


def main(trainingdir, model, num_epochs, size_batch_train, size_batch_test, size_batch_valid, step_metrics, steps_saver, learning_rate, logdir, restore_weights, perform_one_hot, binarize_labels):

    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)

    train_list, valid_list, test_list = get_file_lists(trainingdir)

    train_dataset = create_dataset(filenames=train_list, mode="training", num_epochs=1, batch_size=size_batch_train, perform_one_hot=perform_one_hot, binarize_labels=binarize_labels)
    train_iterator = train_dataset.make_initializable_iterator()
    validation_dataset = create_dataset(filenames=valid_list, mode="validation", num_epochs=1, batch_size=size_batch_valid, perform_one_hot=perform_one_hot, binarize_labels=binarize_labels)
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_dataset = create_dataset(filenames=test_list, mode="testing", num_epochs=1, batch_size=size_batch_test, perform_one_hot=perform_one_hot, binarize_labels=binarize_labels)
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
        logits = model.unet(x, training_placeholder, binarize_labels=binarize_labels)
    elif model == "unet_tensorflow":
        from models import unet_tensorflow as model
        logits = model.unet(x, training=training_placeholder, norm_option=True, binarize_labels=binarize_labels)

    if perform_one_hot:
    #Forward and backprop pass
        if binarize_labels:
            y.set_shape([None, 192, 192, 2])
        else:
            y.set_shape([None, 192, 192, 4])
    else:
        y.set_shape([None, 192, 192, 1])

    y = tf.cast(y, tf.int32)

    if perform_one_hot:
        tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
        loss_op = tf.losses.get_total_loss(name='loss_op')

        #Define the IoU metrics and update operations
        IoU_metrics, IoU_metrics_update = tf.metrics.mean_iou(labels=y, predictions=logits, num_classes=4, name='my_metric_IoU')
        #Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric_IoU")
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    else:
        tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss_op = tf.losses.get_total_loss(name='loss_op')


    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([update_ops, optimizer.minimize(loss_op, global_step=global_step)])

    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'Checkpoint')
    saver = tf.train.Saver()


    ######################################## SUMMARIES #########################################################

    tf.summary.image('input_0', tf.expand_dims(x[:,:,:,0], axis=-1))

    if perform_one_hot:
        tf.summary.image('labels_0', tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,0], axis=-1))
        tf.summary.image('labels_1', tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,1], axis=-1))
        if not binarize_labels:
            tf.summary.image('labels_2', tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,2], axis=-1))
            tf.summary.image('labels_3', tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,3], axis=-1))
        tf.summary.image('prediction_0', tf.expand_dims(logits[:,:,:,0], axis=-1))
        tf.summary.image('prediction_1', tf.expand_dims(logits[:,:,:,1], axis=-1))
        if not binarize_labels:
            tf.summary.image('prediction_2', tf.expand_dims(logits[:,:,:,2], axis=-1))
            tf.summary.image('prediction_3', tf.expand_dims(logits[:,:,:,3], axis=-1))
    else:
        tf.summary.image("labels", tf.cast(y,tf.float32))
        tf.summary.image("prediction", logits[:,:,:,1:])

    #tf.summary.histogram("logits",logits)

    tf.summary.scalar("loss", loss_op)

    summary_op=tf.summary.merge_all()

    # op to write logs to Tensorboard
    logdir = os.path.expanduser(logdir)
    utils.ensure_dir(logdir)
    writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


    ######################################## RUN SESSION #########################################################

    with tf.Session() as sess:

        # Initialize Variables
        if restore_weights:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        train_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        #training, validation and saving
        for epoch in range(num_epochs):
            sess.run(train_iterator.initializer)
            step=0
            try:
                while True:
                    #training
                    #_,cost,summary_val,step_gl = sess.run([train_op,loss_op,summary_op,global_step], feed_dict={handle: train_handle})
                    #writer.add_summary(summary_val,step_gl)

                    #step += 1
                    #print('\n Training step: Epoch {}, batch {} -- Loss: {:.3f}'.format(epoch+1, step+1, cost))


                    #validation
                    if step % step_metrics == 0:
                        total_validation_loss = [] #list where we will store the loss at each batch
                        sess.run(validation_iterator.initializer)
                        step_val=0
                        print('\n Step {}: Saving weights to {}'.format(step+1, model_checkpoint_path))
                        # initialize/reset the running variables of the IoU metrics
                        sess.run(running_vars_initializer)
                        try:
                            while True:
                                cost_valid, _ = sess.run([loss_op, IoU_metrics_update], feed_dict={handle: validation_handle})
                                total_validation_loss.append(cost_valid)
                                step_val += 1
                        except tf.errors.OutOfRangeError:
                            pass
                            #loss
                            total_validation_loss = np.mean(total_validation_loss)
                            validation_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="validation_loss", simple_value=total_validation_loss)])
                            writer.add_summary(validation_loss_summary,step_gl)
                            #IoU metrics
                            IoU_score = sess.run(IoU_metrics)
                        print('\n Epoch {} and training batch {} -- Validation loss {:.3f} and IoU metrics {:.3f}'.format(epoch+1, step,total_validation_loss, IoU_score))


                    #saving
                    if step % STEPS_SAVER == 0:
                        print('\n Step {}: Saving weights to {}'.format(step+1, model_checkpoint_path))
                        saver.save(sess, save_path=model_checkpoint_path, global_step=global_step)

            except tf.errors.OutOfRangeError:
                pass

        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default=TRAININGDIR, help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-btr', '--size_batch_train', type=int, default=BATCH_SIZE_TRAIN, help='Batch size for training')
    parser.add_argument('-bts', '--size_batch_test', type=int, default=BATCH_SIZE_TEST, help='Batch size for testing')
    parser.add_argument('-bval', '--size_batch_valid', type=int, default=BATCH_SIZE_VALID, help='Batch size for validation')
    parser.add_argument('-sm', '--step_metrics', type=int, default=STEP_METRICS, help='frequency of summary within training batches')
    parser.add_argument('-ss', '--steps_saver', type=int, default=STEPS_SAVER, help='frequency of weight saving within training batches')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-r', '--restore_weights', type=float, default=RESTORE_WEIGHTS, help='Restore weights from logdir path.')
    parser.add_argument('-m', '--model', default=MODEL_TO_USE,help='Model to use, either unet_keras or unet_tensorflow.')
    parser.add_argument('-oh', '--perform_one_hot', default=PERFORM_ONE_HOT,help='Perform on-hot encoding for labels.')
    parser.add_argument('-bi', '--binarize_labels', default=BINARIZE_LABELS,help='Perform binarization for labels.')


    args = parser.parse_args()
