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
batch_size = 20000
input_dim = 5

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, input_dim], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    lr = tf.placeholder(tf.float32)
    sw = tf.placeholder(tf.float32,shape=[batch_size],name ='sample_weight')

# add hidden layer-1
l1 = add_layer(xs, input_dim, 100, n_layer=1, activation_function=tf.nn.relu)
# add hidden layer-2
l2 = add_layer(l1, 100, 50, n_layer=2, activation_function=tf.nn.relu)
# add hidden layer-3
l3 = add_layer(l2, 50, 25, n_layer=3, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l3, 25, 1, n_layer=4, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                                        reduction_indices=[1]))
    diff = tf.abs(ys-prediction)
    weighted_diff = tf.multiply(diff,sw)
#    loss = tf.reduce_mean(diff)
    loss = tf.reduce_mean(weighted_diff)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#%%
# traininig process
def training_process(input_data,output_data,IRLS=True,iteration_num=800*10):
    with tf.Session() as sess:        
        init = tf.global_variables_initializer()
        sess.run(init)                
        sn = Nextbacth(np.arange(len(input_data)))
        
        # initialize the sample weights as ones for all data 
        SAMPLE_WEIGHT = np.ones(len(input_data))
       
        
        for i in range(iteration_num):
            # update learninig weight
            max_learning_rate = 0.2
            min_learning_rate = 0.05
            decay_speed = 500.0
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
            # update batch data
            label = sn.next_batch(batch_size)
            # update sample weight
            batch_sample_weight = SAMPLE_WEIGHT[label]
            
            # sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            _,l,a,b= sess.run([train_step,loss,ys,prediction], 
                              feed_dict={xs: input_data[label], 
                                         ys:output_data[label],
                                         lr:learning_rate,
                                         sw:batch_sample_weight})
            
        
            # update sample weight
            if i % 8 == 0 and i!=0:
                print('The %d batch loss is:%f'%(i,l))
                pre = sess.run(prediction,feed_dict={xs: input_data, ys: output_data,lr:learning_rate})
                res = np.abs(pre - output_data)
                res_mean = np.mean(res)
                res_std  = np.std(res)
                
#                if IRLS==True:
#                    for idx,item in enumerate(res):
#                        SAMPLE_WEIGHT[idx] = 1 if item<=res_mean+2*res_std and item>=res_mean-2*res_std else SAMPLE_WEIGHT[idx]*0
                
                if IRLS==True:
                    for idx,item in enumerate(res):
                        SAMPLE_WEIGHT[idx] = np.exp(-item)
    return pre,SAMPLE_WEIGHT,l


#%% L2+None
pre_noise_None,sw_None,loss_L2_None = training_process(input_data=inputs,output_data=outputs[:,1].reshape((-1,1)),IRLS=False,iteration_num=800*100)
