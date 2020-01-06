#!/usr/bin/env python
# coding: utf-8

# In[18]:

#library imports and placeholders for data
import tensorflow as tf
in_msg=tf.placeholder(tf.float32, shape=(b_size, t_size), name='in_msg')
in_key=tf.placeholder(tf.float32, shape=(b_size, k_size), name='in_key')


# In[19]:


def model(collection, msg, key=None):
    #checking to see if its alice bob or eve else case is eve
    if key is not None:
        comb_in = tf.concat(axis=1, values=[msg, key])
    else:
        comb_in=msg
    with tf.variable_scope(collection):
        #creating 32 neuron fully connect net
        fc = tf.layers.dense(comb_in, T_SIZE + K_SIZE, activation=tf.nn.relu)
         #expanded to matrix allow convolution
        fc = tf.expand_dims(fc, 2)
        #first conv [4,1,2]
        conv1=tf.layer.conv1d(fc,kernel_size=4, strides=1, filters=2, padding='SAME', activation=tf.nn.sigmoid)
        #second conv [2,2,4]
        canv2=tf.layer.conv1d(conv1, kernel_size=2, strides=2, filters=4, padding='VALID', activation=tf.nn.sigmoid)
        #third conv [1,4,4]
        conv3=tf.layer.conv1d(conv2, kernel_size=1, strides=4, filters=4, padding='SAME', activation=tf.nn.sigmoid)
        #fourth conv [1,4,1]  and tanh to convert back to (-1,1)
        conv4=tf.layer.conv1d(conv3, kernel_size=1, strides=4, filters=1, padding='SAME', activation=tf.nn.tanh)
        #converting back to a single line binary
        conv4 = tf.squeeze(conv4, 2)
    return conv4

