import tensorflow as tf
from Code.ops import *

#def _phase_shift(I, r):
#    bsize, a, b, c = I.get_shape().as_list()
#    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
#    X = tf.reshape(I, (bsize, a, b, r, r))
#    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
#    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
#    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, b, a*r, r
#    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
#    X = tf.concat(2, [tf.squeeze(x, axis=1) for x in X])  # bsize, a*r, b*r
#    return tf.reshape(X, (bsize, a*r, b*r, 1))
#
#
#def PS(X, r, color=False):
#    if color:
#        Xc = tf.split(3, 3, X)
#        X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
#    else:
#        X = _phase_shift(X, r)
#    return X

def generator(x, scope='g'):
    reuse=tf.AUTO_REUSE   
    with tf.variable_scope(scope, reuse=reuse):
#        output0 = PS(x, 2)
        output1_1 = conv_ln_lrelu(x, 32, '1_1', 'VALID')
        output1_2 = conv_ln_lrelu(output1_1, 32, '1_2', 'SAME')
       
        output2_1 = conv_ln_lrelu(output1_2, 64, '2_1', 'VALID')
        output2_2 = conv_ln_lrelu(output2_1, 64, '2_2', 'SAME')
        
        output3_1 = conv_ln_lrelu(output2_2, 128, '3_1', 'VALID')
        output3_2 = conv_ln_lrelu(output3_1, 128, '3_2', 'SAME')
        
        output4_1 = conv_ln_lrelu(output3_2, 256, '4_1', 'VALID')
        output4_2 = conv_ln_lrelu(output4_1, 256, '4_2', 'SAME')
        
        output5_1 = deconv_ln_relu(output4_2, 128, '1_1', 'VALID')
        output5_2 = tf.concat([output3_2, output5_1], 3)
        output5_3 = conv_ln_lrelu(output5_2, 128, '5_1', 'SAME')
        
        output6_1 = deconv_ln_relu(output5_3, 64, '2_1', 'VALID')
        output6_2 = tf.concat([output2_2, output6_1], 3)
        output6_3 = conv_ln_lrelu(output6_2, 64, '6_1', 'SAME')
        
        output7_1 = deconv_ln_relu(output6_3, 32, '3_1', 'VALID')
        output7_2 = tf.concat([output1_2, output7_1], 3)
        output7_3 = conv_ln_lrelu(output7_2, 32, '7_1', 'SAME')
        
        output8_1 = deconv_ln_relu(output7_3, 1, '4_1', 'VALID')
    return output8_1

def discriminator(x, scope='d', padding='same'):
    reuse=tf.AUTO_REUSE   
    with tf.variable_scope(scope, reuse=reuse):
        outputs1 = tf.layers.conv2d(x, 64, 3, strides=(2, 2), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1', use_bias=False)  
        outputs1 = tf.contrib.layers.instance_norm(outputs1)
        outputs1 = lrelu(outputs1, 0.1)
        
        outputs2 = tf.layers.conv2d(outputs1, 128, 3, strides=(2, 2), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2', use_bias=False)  
        outputs2 = tf.contrib.layers.instance_norm(outputs2)
        outputs2 = lrelu(outputs2, 0.1)
        
        outputs3 = tf.layers.conv2d(outputs2, 256, 3, strides=(2, 2), padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3', use_bias=False)  
        outputs3 = tf.contrib.layers.instance_norm(outputs3)
        outputs3 = lrelu(outputs3, 0.1)
        
        outputs4 = tf.layers.conv2d(outputs3, 512, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4', use_bias=False)  
        outputs4 = tf.contrib.layers.instance_norm(outputs4)
        outputs4 = lrelu(outputs4, 0.1)
        
        outputs5 = tf.layers.conv2d(outputs4, 1, 3, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5', use_bias=False)  
        outputs5 = tf.contrib.layers.instance_norm(outputs5)
        outputs5 = lrelu(outputs5, 0.1)  
    return outputs5

def vgg_model(inputs, scope='vgg'):
    reuse=tf.AUTO_REUSE
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.concat([inputs*255-103.939, inputs*255-116.779, inputs*255-123.68], 3)
        outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_1')
        outputs = tf.layers.conv2d(outputs, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv1_2')
        outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool1')
    
        outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_1')
        outputs = tf.layers.conv2d(outputs, 128, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv2_2')
        outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool2')
        
        outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_1')
        outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_2')
        outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_3')
        outputs = tf.layers.conv2d(outputs, 256, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv3_4')
        outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool3')
    
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_1')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_2')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_3')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv4_4')
        outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2,2), padding='same', name='pool4')
        
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_1')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_2')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_3')
        outputs = tf.layers.conv2d(outputs, 512, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='conv5_4') 
    return outputs