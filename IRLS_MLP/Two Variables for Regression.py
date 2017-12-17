# -*- coding: utf-8 -*-

"""
This code is for IRLS-DNN model test
E-mail: guokai_liu@163.com

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from next import Nextbacth
from mpl_toolkits.mplot3d import Axes3D

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

#%% Make up some real data
    
# make some meshgrid data
sep = 0.05
X = np.arange(-1, 1, sep)
Y = np.arange(-1, 1, sep)
X, Y = np.meshgrid(X, Y)
Z = X**2 + Y**2

# sequencial data
x = X.reshape(-1)
y = Y.reshape(-1)
z = Z.reshape(-1)

# make training data for pure and noisy condition
x_data = np.stack((x,y),axis=1).astype('float32')
y_data = z.reshape((-1,1)).astype('float32')

# make training noisy data
y_data_noise = y_data.copy()
noise_level = 0.1
noise_num = int(1600*noise_level)
noise = np.random.randint(5,10,noise_num)
index = np.random.randint(1,1600,noise_num)
y_data_noise[index]=noise.reshape((-1,1))



# filter the noisy data and left the pure data
fil = y_data==y_data_noise
fil_index = [idx for idx,item in enumerate(fil) if fil[idx]==True]
x_data_pure = x_data[fil_index]
y_data_pure = y_data[fil_index]

#%%
# define placeholder for inputs to network
batch_size = 200
sample_size = len(x_data)
input_dim = 2

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
def training_process(input_data,output_data,IRLS=True,iteration_num=1000):
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
    return pre,SAMPLE_WEIGHT
#%% calculate the prediction under different condtions
pre_pure_IRLS,sw_Pure = training_process(input_data=x_data_pure,output_data=y_data_pure,IRLS=True,iteration_num=10000)
#%%
pre_noise_None,sw_None = training_process(input_data=x_data,output_data=y_data_noise,IRLS=False,iteration_num=10000)
#%%
pre_noise_IRLS,sw_IRLS = training_process(input_data=x_data,output_data=y_data_noise,IRLS=True,iteration_num=10000)

#%% plot the results for analysis
#------------------------------------------------------------------------------
# create a figure
fig = plt.figure(figsize=(18,12))
fig.patch.set_facecolor('white')

# plot the distribution of the ground truth data
ax = fig.add_subplot(231, projection='3d')
ax.scatter(x,y,z)
ax.title.set_text('The Ground Truth')
ax.set_zlim(-1, 10)

# plot the distribution of the noisy data
ax = fig.add_subplot(232, projection='3d')
ax.scatter(x,y,y_data_noise)
ax.title.set_text('The Noisy Data')
ax.set_zlim(-1, 10)

# plot trained model with pure data using IRLS
ax = fig.add_subplot(233, projection='3d')
ax.scatter(x_data_pure[:,0],x_data_pure[:,1],pre_pure_IRLS)
ax.title.set_text('Model with Pure Data')
ax.set_zlim(-1, 10)

# plot trained model with noisy data without using IRLS
ax = fig.add_subplot(234, projection='3d')
ax.scatter(x_data[:,0],x_data[:,1],pre_noise_None)
ax.title.set_text('Model with Noisy Data without using IRLS')
ax.set_zlim(-1, 10)
fig.tight_layout()

# plot trained model with noisy data using IRLS
ax = fig.add_subplot(235, projection='3d')
ax.scatter(x_data[:,0],x_data[:,1],pre_noise_IRLS)
ax.title.set_text('Model with Noisy Data using IRLS')
ax.set_zlim(-1, 10)
fig.tight_layout()
#------------------------------------------------------------------------------