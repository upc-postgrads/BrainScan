import tensorflow as tf
from math import sqrt
import sys



# U-NET MODEL

def unet(data, training=False, norm_option=False, drop_val=0.5,label_output_size=1):

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
    concat1 = tf.concat([conv8,deconv1], axis=-1)
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
    concat2 = tf.concat([conv6,deconv2], axis=-1)
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
    concat3 = tf.concat([conv4,deconv3], axis=-1)
    UpPath_conv5 = tf.layers.conv2d(concat3, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*256))))
    if norm_option == True:
        UpPath_conv5 = tf.layers.batch_normalization(UpPath_conv5)
    UpPath_conv5 = tf.nn.relu(UpPath_conv5)
    UpPath_conv6 = tf.layers.conv2d(UpPath_conv5, 128, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        UpPath_conv6 = tf.layers.batch_normalization(UpPath_conv6, training=training)
    UpPath_conv6 = tf.nn.relu(UpPath_conv6)

    deconv4 = tf.layers.conv2d_transpose(UpPath_conv6, 128, [2,2], strides=[2,2], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        deconv4 = tf.layers.batch_normalization(deconv4)
    deconv4 = tf.nn.relu(deconv4)
    concat4 = tf.concat([conv2,deconv4], axis=-1)
    UpPath_conv7 = tf.layers.conv2d(concat4, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*128))))
    if norm_option == True:
        UpPath_conv7 = tf.layers.batch_normalization(UpPath_conv7)
    UpPath_conv7 = tf.nn.relu(UpPath_conv7)
    UpPath_conv8 = tf.layers.conv2d(UpPath_conv7, 64, [3,3], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
    if norm_option == True:
        UpPath_conv8 = tf.layers.batch_normalization(UpPath_conv8)
    UpPath_conv8 = tf.nn.relu(UpPath_conv8)
    
    
    UpPath_conv9 = tf.layers.conv2d(UpPath_conv8, label_output_size, [1,1], strides=[1,1], padding='SAME', kernel_initializer=tf.initializers.random_normal(stddev=sqrt(2/(3*3*64))))
            
    if label_output_size==1:
        UpPath_conv9_soft = tf.nn.sigmoid(UpPath_conv9)
    else:
        UpPath_conv9_soft = tf.nn.softmax(UpPath_conv9)
        
return UpPath_conv9, UpPath_conv9_soft