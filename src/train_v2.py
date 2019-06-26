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
BATCH_SIZE_VALID = 25
BATCH_SIZE_TEST = 2
STEP_METRICS = 50
LEARNING_RATE = 1e-5
STEPS_SAVER = 100
MODEL_TO_USE = "unet_keras"
RESTORE_WEIGHTS=False
TRAININGDIR = "/path_of_training_images"
LOGDIR = '/tmp'


#########################################################

def count_records(path):

    #function that returns the number of records in a set of TFRecords stored in a directory. We will use it to count the number of
    #training and validation data

    num = 0
    for record_file in os.listdir(path):
        TFRecord_path = os.path.join(path,record_file)
        for record in tf.io.tf_record_iterator(TFRecord_path):
            num += 1
    return num


def main(trainingdir, model, num_epochs, size_batch_train, size_batch_valid, size_batch_test, step_metrics, steps_saver, learning_rate, logdir, restore_weights):

    #param step_valid: after how many batches of training images we perform the validation
    #param step_metrics: after how many batches of training images we keep track of the summary

    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)

    train_images = count_records(os.path.join(trainingdir, 'Training')) #number of training images
    valid_images = count_records(os.path.join(trainingdir, 'Validation')) #number of validation images

    ######################################## DATAFLOW GRAPH #########################################################

    train_list, valid_list, test_list = get_file_lists(trainingdir)

    train_dataset = create_dataset(filenames=train_list,mode="training", num_epochs=num_epochs, batch_size=size_batch_train)
    train_iterator = train_dataset.make_one_shot_iterator()
    validation_dataset = create_dataset(filenames=valid_list,mode="validation", num_epochs=1, batch_size=size_batch_valid)
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_dataset = create_dataset(filenames=test_list,mode="testing", num_epochs=1, batch_size=size_batch_test)
    test_iterator = test_dataset.make_one_shot_iterator()

    # Feedable iterator assigns each iterator a unique string handle it is going to work on
    handle = tf.placeholder(tf.string, shape = [])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    x, y = iterator.get_next()

    if model == "unet_keras":
        from models import unet_keras as model
        x.set_shape([None, 192, 192, 4])
        x = tf.cast(x, tf.float32)
        logits = model.unet(x,True)
    elif model == "unet_tensorflow":
        from models import unet_tensorflow as model
        x.set_shape([None, 192, 192, 4])
        x = tf.cast(x, tf.float32)
        logits = model.unet(x, training=True, norm_option=True)

    #Forward and backprop pass
    y.set_shape([None, 192, 192, 4])
    y = tf.cast(y, tf.int32)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits))
    IoU_metrics = tf.metrics.mean_iou(labels=y, predictions=logits, num_classes=4)
    loss_op = tf.losses.get_total_loss()

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=global_step)

    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'Checkpoint')
    saver = tf.train.Saver()


    ######################################## SUMMARIES #########################################################

    tf.summary.image('input_0',tf.expand_dims(x[:,:,:,0],axis=-1))
    #tf.summary.image("labels",tf.cast(y,tf.float32))
    tf.summary.image('labels_0',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,0],axis=-1))
    tf.summary.image('labels_1',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,1],axis=-1))
    tf.summary.image('labels_2',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,2],axis=-1))
    tf.summary.image('labels_3',tf.expand_dims(tf.cast(y,tf.float32)[:,:,:,3],axis=-1))
    tf.summary.image('prediction_0',tf.expand_dims(logits[:,:,:,0],axis=-1))
    tf.summary.image('prediction_1',tf.expand_dims(logits[:,:,:,1],axis=-1))
    tf.summary.image('prediction_2',tf.expand_dims(logits[:,:,:,2],axis=-1))
    tf.summary.image('prediction_3',tf.expand_dims(logits[:,:,:,3],axis=-1))
    tf.summary.scalar("loss", loss_op)
    tf.summary.histogram("logits",logits)
    summary_op=tf.summary.merge_all()


    ######################################## RUN SESSION #########################################################


    with tf.Session() as sess:

        # Initialize Variables
        if restore_weights:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        # op to write logs to Tensorboard
        logdir = os.path.expanduser(logdir)
        utils.ensure_dir(logdir)
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


        train_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        #training, validation and saving
        for epoch in range(num_epochs):
            for step in range(int(train_images/size_batch_train)):
                #training
                x_aux, y_aux = sess.run([x, y], feed_dict={handle: train_handle})
                _,cost,summary_val,step_gl,logits_val = sess.run([train_op,loss_op,summary_op,global_step,logits], feed_dict={x:x_aux, y:y_aux})
                writer.add_summary(summary_val,step_gl)
                print('\nEpoch {}, batch {} -- Loss: {:.3f}'.format(epoch+1, step+1, cost))

                #validation
                if step % step_metrics == 0:
                    validation_loss = []
                    sess.run(validation_iterator.initializer)
                    for batch_val in range(int(valid_images/size_batch_valid)):
                        x_aux_valid, y_aux_valid = sess.run([x, y], feed_dict={handle: validation_handle})
                        _, cost_valid = sess.run([logits, loss_op], feed_dict={x:x_aux_valid, y:y_aux_valid})
                        validation_loss.append(cost_valid)
                    print('Epoch {} and training batch {} -- Validation loss: {:.3f}'.format(epoch+1, step+1, np.mean(validation_loss)))

                #saving
                if step % steps_saver == 0:
                    print('Step {}\tSaving weights to {}'.format(step+1, model_checkpoint_path))
                    saver.save(sess, save_path=model_checkpoint_path,global_step=global_step)



        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default=TRAININGDIR, help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-btr', '--size_batch_train', type=int, default=BATCH_SIZE_TRAIN, help='Batch size for training')
    parser.add_argument('-bval', '--size_batch_valid', type=int, default=BATCH_SIZE_VALID, help='Batch size for validation')
    parser.add_argument('-bts', '--size_batch_test', type=int, default=BATCH_SIZE_TEST, help='Batch size for testing')
    parser.add_argument('-sm', '--step_metrics', type=int, default=STEP_METRICS, help='frequency of summary within training batches')
    parser.add_argument('-ss', '--steps_saver', type=int, default=STEPS_SAVER, help='frequency of weight saving within training batches')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-r', '--restore_weights', type=float, default=RESTORE_WEIGHTS, help='Restore weights from logdir path.')
    parser.add_argument('-m', '--model', default=MODEL_TO_USE,help='Model to use, either unet_keras or unet_tensorflow.')

    args = parser.parse_args()
