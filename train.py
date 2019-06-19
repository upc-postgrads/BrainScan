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



NUM_EPOCHS = 3
BATCH_SIZE= 1
LEARNING_RATE= 1e-4
STEPS_SAVER = 3

#"zhixuhao"
#"nuria"
# MODEL_TO_USE="nuria"

def get_file_lists(data_dir):
    import glob
    train_list = glob.glob(data_dir + '/Training/' + '*')
    valid_list = glob.glob(data_dir + '/Validation/' + '*')
    test_list = glob.glob(data_dir + '/Testing/' + '*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list, test_list

def train_input_fn(file_path,num_epochs, batch_size):
    return dataset.input_fn(file_path,True,num_epochs,batch_size)

def validation_input_fn(file_path,num_epochs, batch_size):
    return dataset.input_fn(file_path,False,num_epochs,batch_size)

def test_input_fn(file_path,num_epochs, batch_size):
    return dataset.input_fn(file_path,False,num_epochs,batch_size)

#logits: predicció
def loss(labels, logits):
    labels = backend.print_tensor(labels, message='labels = ')
    logits = backend.print_tensor(logits, message='logits = ')

    return tf.losses.sparse_softmax_cross_entropy(labels, logits)
    #return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

    
def main(trainingdir, model, num_epochs, batch_size, learning_rate, logdir, restore_weights):    
    train_list,valid_list, test_list = get_file_lists(trainingdir)
    next_batch = train_input_fn(train_list,num_epochs, batch_size)
    test_batch = test_input_fn(test_list,num_epochs, batch_size)
    # tf Graph Input
    x = tf.placeholder(tf.float32, shape=[None, 192, 192, 4], name='x')
    y = tf.placeholder(tf.int32, shape=[None, 192, 192, 1], name='y')


    if model == "zhixuhao":
        import model as model
        y_ = model.unet(x,True)
    elif model == "nuria":
        import unet
        y_ = unet.unet_model(x, training=True)


    loss = tf.reduce_mean(loss(y, y_))
    #loss = tf.losses.get_total_loss()
    #loss = tf.reduce_mean(loss(labels=y,logits=y_))

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

    train_op = optimizer.minimize(loss)

    
    # Weight saver
    model_checkpoint_path = os.path.join(logdir, 'BrainScan')
    saver = tf.train.Saver()
    
    
    #prediction = tf.argmax(y_, 1)
    with tf.Session() as sess:

        if restore_weights:
            saver.restore(sess, restore_weights)
        else:
            sess.run(tf.global_variables_initializer())
        
        # train
        # sess.run(tf.global_variables_initializer())

        # op to write logs to Tensorboard
        logdir = os.path.expanduser(logdir)
        utils.ensure_dir(logdir)
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())


        for epoch in range(num_epochs):
        
        
            x_train_batch, y_train_batch = sess.run(next_batch)
            current_loss, _ = sess.run([loss, train_op], feed_dict={x:x_train_batch,
                                                                     y:y_train_batch})
            print(current_loss)
            
            if epoch % STEPS_SAVER == 0:
                print('Step {}\tSaving weights to {}'.format(model_checkpoint_path))
                saver.save(sess, save_path=model_checkpoint_path)
            
            

        try:
            y_pred = []

            x_test_batch = sess.run(test_batch)
            sess.run([y_pred],feed_dict={x:x_test_batch})
            print(y_pred)
        except tf.errors.OutOfRangeError:
            print("u fucked up")

        """
        #Les dades son les correctes:
            for j in range(x_train_batch.shape[0]):
                outputFile=os.path.join("/home/deivit/Desktop/dades/Documents/david/upc/AIDL/projecte/unet/BrainTumourImages/caca",str(epoch) + str(j)  +  ".jpg" )
                data=x_train_batch[j,:,:,0]
                #data = data.reshape(192, 192, 4)
                im=Image.fromarray(data)
                im.save(outputFile)
            """
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-t', '--trainingdir', default='../BrainTumourImages/Generated/', help='Location of the TFRecors for training')
    parser.add_argument('-l', '--logdir', default='/tmp/aidl', help='Log dir for tfevents')
    parser.add_argument('-e', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-r', '--restore', help='Path to model checkpoint to restore weights from.')
    parser.add_argument('-m', '--model', default="nuria",help='Model to use, either zhixuhao or nuria.')
    
    args = parser.parse_args()
    
    
    main(args.trainingdir,args.model, args.num_epochs, args.batch_size, args.learning_rate, args.logdir, args.restore)

    
    
    
