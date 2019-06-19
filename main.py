import tensorflow as tf
from Generate_Batch import *
from unet import *
import os

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

#training
train_images = count_records('C:/Users/nuria/Desktop/FinalProject/BrainTumour/Generated_TFRecords/Training')
#validation
valid_images = count_records('C:/Users/nuria/Desktop/FinalProject/BrainTumour/Generated_TFRecords/Validation')

#########################################################

#Training parameters
size_batch_train = 25
size_batch_valid = int(valid_images/int(train_images/size_batch_train))
n_epochs = 2
learning_rate = 1e-6


######################################## DATAFLOW GRAPH #########################################################

train_list, valid_list = get_file_lists("C:/Users/nuria/Desktop/FinalProject/BrainTumour/Generated_TFRecords")
batch_train = input_fn(filenames=train_list, mode='training', num_epochs=n_epochs, batch_size=size_batch_train)
batch_valid = input_fn(filenames=valid_list, mode='validation', num_epochs=n_epochs, batch_size=size_batch_valid)

# Graph inputs
x = tf.placeholder('float', shape=[None, 192, 192, 4], name='x')
y = tf.placeholder('float', shape=[None, 192, 192, 1], name='y')

#Forward and backprop pass
logits = unet_model(x, training=True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

######################################## RUN SESSION #########################################################

with tf.Session() as sess:

    # Initialize Variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #training
    for epoch in range(n_epochs):
        for step in range(int(train_images/size_batch_train)):

            #training
            batch_images, batch_labels = sess.run(batch_train)
            _,cost = sess.run([train_op,loss], feed_dict={x:batch_images, y:batch_labels})

            #validation
            batch_images_valid, batch_labels_valid = sess.run(batch_valid)
            cost_valid = sess.run(loss, feed_dict={x:batch_images_valid, y:batch_labels_valid})

            print('\nEpoch {}, batch {} -- Loss: {:.3f}, Validation Loss: {:.3f}'.format(epoch+1, step+1, cost, cost_valid))
