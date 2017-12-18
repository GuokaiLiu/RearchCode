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

#%% Parameters Setting
sep = 0.05    
noise_level = 0.15
noise_mean = 0
noise_std = 10
#learning rate 
max_learning_rate = 0.2
min_learning_rate = 0.01
decay_speed = 500.0
# iteration number
iter_num = 5000

#%% make pure data
x = np.arange(-4, 4, sep).reshape(-1,1)
y = x**2

#%%
# make noisy data
noise_num = int(len(x)*noise_level)
#noise = np.random.normal(5,10,noise_num).reshape((-1,1))
noise = np.random.normal(noise_mean,noise_std,noise_num).reshape((-1,1))
index = np.random.choice(len(x),noise_num,replace = False).reshape((-1,1))



y_noise = y.copy()
noisy_data = noise+y[index.reshape(-1)]
y_noise[index.reshape(-1)]=noisy_data
x_data = x
y_data = y.copy()
y_data[index.reshape(-1)]=noise+y[index.reshape(-1)]

#%%
# filter the noisy data and left the pure data
def fil_index(y_t,y_n,flag='Noise'):
    if flag=='True':
        fil = y_t==y_n
        fil_index = [idx for idx,item in enumerate(fil) if fil[idx]==True]
    elif flag=='Noise':
        fil = y_t!=y_n
        fil_index = [idx for idx,item in enumerate(fil) if fil[idx]==True]
    else:
        print('Wrong flag')
    return fil_index

index_true = fil_index(y,y_noise,flag='True')
index_noise = fil_index(y,y_noise,flag='Noise')

x_p = x[index_true]
y_p = y[index_true]
x_n = x[index_noise]
y_n = y_noise[index_noise]



#%%
# define placeholder for inputs to network
batch_size = 20
sample_size = len(x)
input_dim = 1

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, input_dim], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    lr = tf.placeholder(tf.float32)
    sw = tf.placeholder(tf.float32,shape=[batch_size],name ='sample_weight')

# add hidden layer-1
l1 = add_layer(xs, input_dim, 10, n_layer=1, activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l1,10 , 1, n_layer=4, activation_function=None)

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
def training_process(input_data,output_data,true_y,IRLS=True,iteration_num=1000):
    with tf.Session() as sess:        
        init = tf.global_variables_initializer()
        sess.run(init)                
        sn = Nextbacth(np.arange(len(input_data)))
        
        # initialize the sample weights as ones for all data 
        SAMPLE_WEIGHT = np.ones(len(input_data))
       
        
        for i in range(iteration_num):
            # update learninig weight

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
                ave_true_loss = np.mean(np.abs(pre-true_y))
                
                if IRLS==True:
                    for idx,item in enumerate(res):
                        SAMPLE_WEIGHT[idx] = 1 if item<=res_mean+res_std and item>=res_mean-res_std else SAMPLE_WEIGHT[idx]*0
                
#                if IRLS==True:
#                    for idx,item in enumerate(res):
#                        SAMPLE_WEIGHT[idx] = np.exp(-item)
            
    return pre,SAMPLE_WEIGHT,ave_true_loss
#%% calculate the prediction under different condtions
#%%
pre_pure_IRLS,sw_Pure,loss_pu_IRLS = training_process(input_data=x_p,output_data=y_p,true_y=y_p,IRLS=True,iteration_num=iter_num)
#%%
pre_noise_None,sw_None,loss_sw_None = training_process(input_data=x_data,output_data=y_data,true_y=y,IRLS=False,iteration_num=iter_num)
#%%
pre_noise_IRLS,sw_IRLS,loss_sw_IRLS = training_process(input_data=x_data,output_data=y_data,true_y=y,IRLS=True,iteration_num=iter_num)

#%% plot the results for analysis
#------------------------------------------------------------------------------
# create a figure
fig = plt.figure(figsize=(6,6))
fig.patch.set_facecolor('white')
ps = 5
# plot the distribution of the ground truth data
ax = fig.add_subplot(411)
plt.scatter(x,y,s=ps)
plt.title('The ground truth data: $y=x^2$')
# plot the distribution of the ground truth data
ax = fig.add_subplot(412)
plt.scatter(x_n,y[index_noise],c='r',hold='True',s=ps)
plt.scatter(x_p,y_p,s=ps)
plt.title('The selected noisy positions')
# plot the distribution of the noisy data
ax = fig.add_subplot(413)
plt.scatter(x_n,y_n,c='r',hold='True',s=ps)
plt.scatter(x_p,y_p,s=ps)
plt.title('The noisy data distribution')
# plot the training data
ax = fig.add_subplot(414)
plt.scatter(x_data,y_data,hold='True',s=ps)
plt.title('The training data')

fig.tight_layout()
#------------------------------------------------------------------------------
#%%
fig = plt.figure(figsize=(6,6))
fig.patch.set_facecolor('white')
ps = 5

# plot model trained with noisy data and without IRLS
ax = fig.add_subplot(311)
plt.scatter(x,pre_noise_None,c='r',s=3,label='predict')
plt.scatter(x_data,y_data,c='b',s=3,label='train')
plt.legend(loc='best')
plt.title('Training without IRLS loss: %f'%loss_sw_None)

# plot model trained with noisy data and IRLS
ax = fig.add_subplot(312)
plt.scatter(x,pre_noise_IRLS,c='r',s=3,label='predict')
plt.scatter(x_data,y_data,c='b',s=3,label='train')
plt.legend(loc='best')
plt.title('Training with IRLS loss: %f'%loss_sw_IRLS)

# plot model trained with pure data and IRLS
ax = fig.add_subplot(313)
plt.scatter(x_p,pre_pure_IRLS,c='r',s=3,label='predict')
plt.scatter(x_p,y_p,c='b',s=3,label='train')
plt.legend(loc='best')
plt.title('Training with IRLS loss using pure data: %f'%loss_pu_IRLS)

fig.tight_layout()
plt.savefig('dnn-IRLS-result.jpg')

#%%
#s_index = []
#s_num = 0
#for idx, item in enumerate(sw_IRLS):
#    if item==0:
#        s_num = s_num+1
#        s_index.append(idx)
#
#outlier = np.zeros((2,len(index))) 
#outlier[0,:]=np.sort(index)
#outlier[1,0:len(s_index)] = s_index
