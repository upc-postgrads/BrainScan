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
NUM_EPOCHS = 1
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_TEST = 2
BATCH_SIZE_VALID = 25
STEP_VALID = 50
STEP_METRICS = 50
LEARNING_RATE = 1e-4
STEPS_SAVER = 100
MODEL_TO_USE = "unet_keras"


TRAININGDIR = "C:/Users/Eduard/Desktop/BrainTumorImages/Generated/"
LOGDIR = "C:/Users/Eduard/Desktop/BrainTumorImages/Generated/logs/"
#David
#TRAININGDIR = "../BrainTumourImages/Generated/"
#LOGDIR = '/tmp/aidl'


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

    return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)

def loss_rara2(labels, logits):
    #https://github.com/tensorflow/tensorflow/issues/10021
    #https://stackoverflow.com/questions/40198364/how-can-i-implement-a-weighted-cross-entropy-loss-in-tensorflow-using-sparse-sof/46984951#46984951
    class_weights = tf.constant([0.1 , 0.3 , 0.3 , 0.3])  # 3 classes
    sample_weights = tf.gather(class_weights, labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits,weights=sample_weights)

def loss_sparse(labels, logits):
    #labels = backend.print_tensor(labels, message='labels = ')
    #logits = backend.print_tensor(logits, message='logits = ')
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)






def main(trainingdir, model, num_epochs, size_batch_train, size_batch_test, size_batch_valid, step_valid, step_metrics, steps_saver, learning_rate, logdir, restore_weights):

    #param step_valid: after how many batches of training images we perform the validation
    #param step_metrics: after how many batches of training images we keep track of the summary

    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)

    train_images = count_records(os.path.join(trainingdir, 'Training')) #number of training images
    valid_images = count_records(os.path.join(trainingdir, 'Validation')) #number of validation images

    ######################################## DATAFLOW GRAPH #########################################################

    train_list, valid_list, test_list = get_file_lists(trainingdir)
    batch_train = input_fn(filenames=train_list,mode="training", num_epochs=num_epochs, batch_size=size_batch_train)
    batch_valid = input_fn(filenames=valid_list, mode='validation', num_epochs=num_epochs, batch_size=size_batch_valid)
    batch_test = input_fn(filenames=test_list, mode='testing', num_epochs=num_epochs, batch_size=size_batch_test)

    # Graph inputs
    x = tf.placeholder('float', shape=[None, 192, 192, 4], name='x')
    y = tf.placeholder('int32', shape=[None, 192, 192, 4], name='y')
    #y = tf.one_hot(indices=tf.squeeze(y), depth=4)


    tf.summary.image("input_0",tf.expand_dims(x[:,:,:,0],axis=-1))
    tf.summary.image("labels",tf.cast(y,tf.float32))
    #y=tf.squeeze(y)


    if model == "unet_keras":
        from models import unet_keras as model
        logits = model.unet(x,True)
    elif model == "unet_tensorflow":
        from models import unet_tensorflow as model
        logits = model.unet(x, training=True, norm_option=True)

    tf.summary.image("prediction", logits[:,:,:,1:])
    tf.summary.histogram("logits",logits)
    #Forward and backprop pass
    loss= tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits))
    tf.summary.scalar("loss", loss)
    IoU_metrics = tf.metrics.mean_iou(labels=y, predictions=logits, num_classes=4)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)

    # Weight saver
    # model_checkpoint_path = os.path.join(logdir, 'Checkpoint/')
    saver = tf.train.Saver()

    ######################################## RUN SESSION #########################################################
    summary_op=tf.summary.merge_all()

    with tf.Session() as sess:
        
        # Initialize Variables
        if restore_weights:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            sess.run(tf.global_variables_initializer())   
            sess.run(tf.local_variables_initializer())
      
        # op to write logs to Tensorboard
        logdir = os.path.expanduser(args.logdir)
        utils.ensure_dir(logdir)
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        #training, validation and saving
        for epoch in range(num_epochs):
            # for step in range(int(train_images/size_batch_train)):
            for step in range(3):

                #training
                batch_images, batch_labels = sess.run(batch_train)
                _,cost = sess.run([train_op,loss], feed_dict={x:batch_images, y:batch_labels})
                print('/nEpoch {}, batch {} -- Loss: {:.3f}'.format(epoch+1, step+1, cost))

                if step % step_metrics == 0:
                    summary_val,step_gl,logits_val = sess.run([summary_op,global_step,logits], feed_dict={x:batch_images, y:batch_labels})
                    writer.add_summary(summary_val,step_gl)


                #validation
                cost_validation = []
                IoU_validation = []
                
                
                # if step % step_valid == 0:
                    # for batch in range(int(valid_images/size_batch_valid)):
                        # batch_images_valid, batch_labels_valid = sess.run(batch_valid)
                        # cost_valid = sess.run(loss, feed_dict={x:batch_images_valid, y:batch_labels_valid})
                        # cost_validation.append(cost_valid)
                        # IoU = sess.run(IoU_metrics, feed_dict={x:batch_images_valid, y:batch_labels_valid})
                        # IoU_validation.append(IoU)
                    # print('/nEpoch {} -- Validation Loss: {:.3f} and IoU Metrics: {:.3f}'.format(epoch+1, np.mean(cost_validation), np.mean(IoU_validation)))



                #saving
                if step % steps_saver == 0:
                    print('Step {}/tSaving weights to {}'.format(step+1, logdir))
                    saver.save(sess, save_path=logdir,global_step=global_step)

    #Predictions
        try:
            tf.summary.image("output", logits[:,:,:,1:])
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
            print ((np.array(batch_test)).shape)
            x_test_batch = sess.run(batch_test)
            print ((np.array(x_test_batch)).shape)
            sess.run(logits,feed_dict={x:x_test_batch})
            print(logits)

        except tf.errors.OutOfRangeError:
            print("u fucked up")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default=TRAININGDIR, help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-btr', '--size_batch_train', type=int, default=BATCH_SIZE_TRAIN, help='Batch size for training')
    parser.add_argument('-bts', '--size_batch_test', type=int, default=BATCH_SIZE_TEST, help='Batch size for testing')
    parser.add_argument('-bval', '--size_batch_valid', type=int, default=BATCH_SIZE_VALID, help='Batch size for validation')
    parser.add_argument('-sv', '--step_valid', type=int, default=STEP_VALID, help='frequency of validation within training batches')
    parser.add_argument('-sm', '--step_metrics', type=int, default=STEP_METRICS, help='frequency of summary within training batches')
    parser.add_argument('-ss', '--steps_saver', type=int, default=STEPS_SAVER, help='frequency of weight saving within training batches')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-r', '--restore', help='Path to model checkpoint to restore weights from.')
    parser.add_argument('-m', '--model', default=MODEL_TO_USE,help='Model to use, either unet_keras or unet_tensorflow.')

    args = parser.parse_args()


main(args.trainingdir,args.model, args.num_epochs, args.size_batch_train, args.size_batch_test, args.size_batch_valid, args.step_valid, args.step_metrics, args.steps_saver, args.learning_rate, args.logdir, args.restore)
