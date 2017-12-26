# -*- coding: utf-8 -*-

"""
This code is for IRLS-DNN model test
E-mail: guokai_liu@163.com

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import math
from next import Nextbacth
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#%% define the dense layer
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size],stddev=0.1), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]), name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
#%%
# define placeholder for inputs to network
batch_size = 200
input_dim = 5
max_learning_rate = 0.01
min_learning_rate = 0.01
decaprespeed = 500.0


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, input_dim], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    lr = tf.placeholder(tf.float32)
with tf.name_scope('k_value'):    
   k = tf.placeholder(tf.float32,name ='tunning_constant')

# add hidden layer-1
l1 = add_layer(xs, input_dim, 100, n_layer=1, activation_function=tf.nn.relu)
# add hidden layer-2
l2 = add_layer(l1, 100, 50, n_layer=2, activation_function=tf.nn.relu)
# add hidden layer-3
l3 = add_layer(l2, 50, 25, n_layer=3, activation_function=tf.nn.relu)
# add output layer
pre = add_layer(l3, 25, 1, n_layer=4, activation_function=tf.nn.sigmoid)


#%%
# traininig process
def training_process(train_x,train_y,iteration_num=1,loss_fun='Bisquare'):
    
    with tf.name_scope('loss'):
        if loss_fun=='L2':
            loss = tf.reduce_sum(tf.pow(pre-ys,2)) / (2*batch_size)
        elif loss_fun=='L1':
            loss = tf.reduce_sum(tf.abs(pre-ys)) / (2*batch_size)
        elif loss_fun=='Huber':
            loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(pre-ys),k),        # condition
                                  (k*tf.abs(pre-ys)-0.5*k**2)/batch_size,              # plan A
                                  0.5*tf.pow(pre-ys,2)                         # plan B
                                 ))/ (2*batch_size)                                    # sum and average
        elif loss_fun=='Bisquare':
            loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(pre-ys),k),        # condition
                                  k*1./6*tf.ones_like(pre),                    # plan A
                                  k*1./6*(1-(1-((pre-ys)*1./k)**2)**3)         # plan B
                                 ))/ (2*batch_size)                                    # sum and average
        else:
            print('Wrong Loss function')
    
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
   
    #%%
    with tf.Session() as sess:    
        tf.summary.FileWriter('logs/',sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)                
        
#------------------------------------------------------------------------------     
        sn = Nextbacth(np.arange(len(train_x)))
        k_=1*4.685
        
        for i in range(iteration_num):
            # update learninig weight

            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decaprespeed)
            # update batch data
            label = sn.next_batch(batch_size)
            batch_x = train_x[label]
            batch_y = train_y[label]
            # sess.run(train_step, feed_dict={xs: x_data, ys: predata})
            _,loss_,pre_= sess.run([train_step,loss,pre], 
                                  feed_dict={xs: batch_x, 
                                             ys: batch_y,
                                             lr: learning_rate,
                                             k:  k_})
            mar_error =np.median(np.abs(pre_-batch_y.reshape(-1)))
            k_ = mar_error*4.685
            print('The %d th loss_:%f'%(i,loss_))
            y_p = sess.run([pre],feed_dict={xs:train_x})
    print(iteration_num)
    return y_p
#------------------------------------------------------------------------------
#%%
from earthquake_0_preparation import inputs, outputs
final_pre= training_process(train_x=inputs,train_y=outputs,iteration_num=100,loss_fun='Bisquare')
#%%
plt.figure(num=3)
final_pre2 = final_pre[0]
plt.hist(final_pre2)

