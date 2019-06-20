import os
import argparse
import tensorflow as tf
from dataset import *
import utils
from tensorflow.python.keras import backend as K


# Training parameters
NUM_EPOCHS = 1
BATCH_SIZE= 25
LEARNING_RATE= 1e-4
STEPS_SAVER =100
MODEL_TO_USE="zhixuhao"
TRAININGDIR="/Users/aitorjara/Desktop//BrainTumourImages/Generated/"
LOGDIR='/tmp/aidl'

#########################################################

#number of training and validation data

#function that returns the number of records in a set of TFRecords stored a directory
def count_records(path):
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

# optimizer


    
 


def main(trainingdir, model, num_epochs, batch_size, learning_rate, logdir, restore_weights): 

    
    global_step=tf.get_variable('global_step',dtype=tf.int32,initializer=0,trainable=False)    
    #training
    train_images = count_records(os.path.join(trainingdir, 'Training'))
    #validation
    valid_images = count_records(os.path.join(trainingdir, 'Validation'))

    size_batch_valid = int(valid_images/int(train_images/batch_size))
    
    ######################################## DATAFLOW GRAPH #########################################################
    
    train_list, valid_list, test_list = get_file_lists(trainingdir)
    batch_train = input_fn(filenames=train_list,mode="training", num_epochs=num_epochs, batch_size=batch_size)
    batch_valid = input_fn(filenames=valid_list, mode='validation', num_epochs=num_epochs, batch_size=size_batch_valid)
    batch_test = input_fn(filenames=test_list, mode='testing', num_epochs=num_epochs, batch_size=size_batch_valid)
        
    # Graph inputs
    x = tf.placeholder('float', shape=[None, 192, 192, 4], name='x')
    y = tf.placeholder('int32', shape=[None, 192, 192, 1], name='y')

    tf.summary.image("input_0",tf.expand_dims(x[:,:,:,0],axis=-1))
    tf.summary.image("labels",tf.cast(y,tf.float32))
    #y=tf.squeeze(y)
    
    
    if args.model == "zhixuhao":
        import model as model
        logits = model.unet(x,True)
    elif args.model == "nuria":
        import unet         
        logits = unet.unet_model(x, training=True, norm_option=True)        
        
    tf.summary.image("prediction", logits[:,:,:,1:])
    tf.summary.histogram("logits",logits)
    #Forward and backprop pass
    loss= tf.reduce_mean(loss_sparse(labels=y, logits=logits))
    tf.summary.scalar("loss", loss)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss,global_step=global_step)
    
    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'Checkpoint')
    saver = tf.train.Saver()
    
    ######################################## RUN SESSION #########################################################
    summary_op=tf.summary.merge_all()
    
    with tf.Session() as sess:
    
        # Initialize Variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        # op to write logs to Tensorboard
        logdir = os.path.expanduser(args.logdir)
        utils.ensure_dir(logdir)
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())         
        
        if restore_weights:
            saver.restore(sess, restore_weights)
        else:
            sess.run(tf.global_variables_initializer())       
    
        #training, validation and saving
        for epoch in range(num_epochs):
            for step in range(int(train_images/batch_size)):
    
                #training
                batch_images, batch_labels = sess.run(batch_train)
                _,cost,summary_val,step,logits_val = sess.run([train_op,loss,summary_op,global_step,logits], feed_dict={x:batch_images, y:batch_labels})
                writer.add_summary(summary_val,step)
                #validation
                batch_images_valid, batch_labels_valid = sess.run(batch_valid)
                cost_valid = sess.run(loss, feed_dict={x:batch_images_valid, y:batch_labels_valid})
    
                print('\nEpoch {}, batch {} -- Loss: {:.3f}, Validation Loss: {:.3f}'.format(epoch+1, step+1, cost, cost_valid))
                
                if step % STEPS_SAVER == 0:
                    print('Step {}\tSaving weights to {}'.format(step+1, model_checkpoint_path))
                    saver.save(sess, save_path=model_checkpoint_path,global_step=global_step)
                    
    #Predictions
        try:
            y_pred = []
            x_test_batch = sess.run(batch_test)
            sess.run([y_pred],feed_dict={x:x_test_batch})
            print(y_pred)
    
        except tf.errors.OutOfRangeError:
            print("u fucked up")                    
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default=TRAININGDIR, help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default=LOGDIR, help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-r', '--restore', help='Path to model checkpoint to restore weights from.')
    parser.add_argument('-m', '--model', default=MODEL_TO_USE,help='Model to use, either zhixuhao or nuria.')
    
    args = parser.parse_args()
    
    
main(args.trainingdir,args.model, args.num_epochs, args.batch_size, args.learning_rate, args.logdir, args.restore)            