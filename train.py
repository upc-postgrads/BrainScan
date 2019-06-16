#https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
#https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
#http://androidkt.com/train-keras-model-with-tensorflow-estimators-and-datasets-api/
#https://github.com/Tony607/Keras_catVSdog_tf_estimator/blob/master/keras_estimator_vgg16-cat_vs_dog.ipynb
#http://androidkt.com/feeding-your-own-data-set-into-the-cnn-model-in-tensorflow/


#data augmentation
#https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb#scrollTo=tkNqQaR2HQbd
#http://androidkt.com/tensorflow-image-augmentation-using-tf-image/

#watch -n0.1 nvidia-smi

import tensorflow as tf
import os
import utils
from tensorflow.keras import backend
#from PIL import Image
import argparse
#from tensorflow.keras.preprocessing import image
import dataset



NUM_EPOCHS = 10
BATCH_SIZE= 10
LEARNING_RATE= 1e-4

#"zhixuhao"
#"nuria"
MODEL_TO_USE="nuria"

def get_file_lists(data_dir):
    import glob
    train_list = glob.glob(data_dir + '/Training/' + '*')
    valid_list = glob.glob(data_dir + '/Validation/' + '*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list

def train_input_fn(file_path,num_epochs, batch_size):
    return dataset.input_fn(file_path,True,num_epochs,batch_size)

def validation_input_fn(file_path,num_epochs, batch_size):
    return dataset.input_fn(file_path,False,num_epochs,batch_size)

#logits: predicció
def loss(labels, logits):
    labels = backend.print_tensor(labels, message='labels = ')
    logits = backend.print_tensor(logits, message='logits = ')    
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)
    #return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pipeline execution')   
    parser.add_argument('-t', '--trainingdir', default='../BrainTumourImages/Generated/', help='Location of the TFRecors for training')     
    parser.add_argument('-l', '--logdir', default='/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')

    args = parser.parse_args()    
    train_list,valid_list = get_file_lists(args.trainingdir)
    next_batch = train_input_fn(train_list,args.num_epochs, args.batch_size)
    
    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, 192, 192, 4], name='x')
    y = tf.placeholder(tf.int32, shape=[None, 192, 192, 1], name='y')
    

    if MODEL_TO_USE == "zhixuhao":
        import model as model
        y_ = model.unet(x,True)
    elif MODEL_TO_USE == "nuria":
        import unet         
        y_ = unet.unet_model(x, training=True)        


    loss = tf.reduce_mean(loss(y, y_))
    #loss = tf.losses.get_total_loss()
    #loss = tf.reduce_mean(loss(labels=y,logits=y_))    
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    
    train_op = optimizer.minimize(loss)
    
    #prediction = tf.argmax(y_, 1)    
    with tf.Session() as sess:
        
        # train
        sess.run(tf.global_variables_initializer())
        
        # op to write logs to Tensorboard
        logdir = os.path.expanduser(args.logdir)
        utils.ensure_dir(logdir)
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())        
        
        
        for epoch in range(args.num_epochs):
            x_train_batch, y_train_batch = sess.run(next_batch)
            current_loss, _ = sess.run([loss, train_op], feed_dict={x:x_train_batch,
                                                                     y:y_train_batch})
            print(current_loss)
            
            """
            #Les dades son les correctes:
            for j in range(x_train_batch.shape[0]):
                outputFile=os.path.join("/home/deivit/Desktop/dades/Documents/david/upc/AIDL/projecte/unet/BrainTumourImages/caca",str(epoch) + str(j)  +  ".jpg" )
                data=x_train_batch[j,:,:,0]
                #data = data.reshape(192, 192, 4)            
                im=Image.fromarray(data)
                im.save(outputFile)   
            """  
