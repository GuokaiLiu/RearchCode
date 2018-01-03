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
from utils.next import Nextbacth
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#%% plot the robust result 
def compare_plot(data_true,data_pred,name_idx,loss_name,loss_val):
    plt.ioff()
    plt.figure(figsize=(15,9))
    ax =plt.subplot(111)
    fig = plt.gcf()
    mpl.rcParams.update({'font.size': 12})
    fig.patch.set_facecolor('white')
    plt.scatter(np.arange(len(data_pred)),data_pred,c='b',s=0.5,edgecolors='b',label='Robust Regression')   
    plt.scatter(np.arange(len(data_true)),data_true,c='r',s=0.5,edgecolors='r',label='faaut')
    ax.grid()
    ax.set_xlabel('number of points')
    ax.set_ylabel("faaut('r') vs Robust Regreesion('b')")
    ax.legend()
    ax.set_title('Robust Regression with '+loss_name+' Loss: '+str(loss_val))
    plt.savefig('data/plot4/'+name_idx+'.png')
#    plt.savefig('data/plot4/'+loss_name+str(name_idx)+'.png',dpi=180)
    
    plt.close(fig)
#%%
# seperate the uncertain and certain data index
def cu_idx(data):
    L=  len(data)
    uct_idx = np.where(data['faaut']<0)[0]         # uncertain index 
    uct_pct = len(uct_idx)*1.0/L  # uncertain percentage
    _idx_all = np.arange(L)       
    cet_idx = np.setxor1d(_idx_all,uct_idx) # certain index
    return cet_idx,uct_idx,uct_pct

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
batch_size = 240
input_dim = 5
max_learning_rate = 0.2
#min_learning_rate = 0.0005
min_learning_rate = 0.001
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
pre = add_layer(l3, 25, 1, n_layer=4, activation_function=None)


#%%
# traininig process
def training_process(train_x,train_y,
                     pre_data_x,pre_data_y,
                     iteration_num=1,
                     loss_fun='Bisquare'):
    
    with tf.name_scope('loss'):
        if loss_fun=='L2':
            loss = tf.reduce_sum(tf.pow(pre-ys,2)) / (2*batch_size)
        elif loss_fun=='L1':
            loss = tf.reduce_sum(tf.abs(pre-ys)) / (2*batch_size)
        elif loss_fun=='Huber':
#            loss = tf.losses.huber_loss(ys,pre)
            loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(pre-ys),k),        # condition
                                  (k*tf.abs(pre-ys)-0.5*k**2)/batch_size,      # plan A
                                  0.5*tf.pow(pre-ys,2)                         # plan B
                                 ))/ (2*batch_size)                            # sum and average
            
            
        elif loss_fun=='Bisquare':
            loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(pre-ys),k),        # condition
                                  k*1./6*tf.ones_like(pre),                    # plan A
                                  k*1./6*(1-(1-((pre-ys)*1./k)**2)**3)         # plan B
                                 ))/ (2*batch_size)                            # sum and average
        else:
            print('Wrong Loss function')
    
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    
   
    #%%
    with tf.Session() as sess:    
#        tf.summary.FileWriter('logs/',sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)                
        
#------------------------------------------------------------------------------     
        sn = Nextbacth(np.arange(len(train_x)))
        k_=1*4.685
        
        min_num = 5
        early_stop = np.ones(min_num)*10
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
#            print('The %d th loss_:%f'%(i,loss_))
            
            min_val = min(early_stop)
            early_stop[i%min_num]=loss_
            if i%50 ==0:
                print('K=%f'%k_)
                print('Epoch----',i)
#            if i>500 and early_stop[i%min_num]>=min_val:
#                print('early_stop_array:/n',early_stop)
#                print('minimum:/n',min(early_stop))
#                print('current:/n',early_stop[i%5])
#                break
            
#            y_p = sess.run([pre],feed_dict={xs:train_x})
#    print(iteration_num)
             
            loss_,pre_ = sess.run([loss,pre],feed_dict={xs:pre_data_x,ys:pre_data_y,k:k_})
#            if i%100==0:
#                compare_plot(pre_data_y,
#                 pre_,
#                 name_idx=i,
#                 loss_name=loss_fun,
#                 loss_val=loss_)
        loss_,pre_ = sess.run([loss,pre],feed_dict={xs:pre_data_x,ys:pre_data_y,k:k_})   
    return loss_,pre_,i
#------------------------------------------------------------------------------
#%% train and predict for splited data
for i in np.arange(347):
    split = pd.read_pickle('data/split/train_'+str(i+1)+'.pkl')
    idx_cet,idx_uct,uct_pct = cu_idx(split)
    c_data = split.iloc[idx_cet]                # select the certain data
    p_data_x = split.iloc[:,0:5].as_matrix()    # test with all splited data x
    p_data_y = split.iloc[:,5].reshape((-1,1))  # test with all splited data y
    inputs = c_data.iloc[:,0:5].as_matrix()     # train with selected certain data x
    outputs = c_data.iloc[:,5].reshape((-1,1))  # train with selected certain data y
    
    loss_f = 'L1'                          # ['L2','L1','Bisquare','Huber']
    
    t1 = time.time()
    tt_loss_,fn_pre,iter_num= training_process(train_x=inputs,
                                      train_y=outputs,
                                      pre_data_x = p_data_x,
                                      pre_data_y = p_data_y,
                                      iteration_num=2000,
                                      loss_fun=loss_f)
    np.save('data/loss/loss_'+str(i+1)+'.npy',tt_loss_)
    np.save('data/pred/pred_'+str(i+1)+'.npy',fn_pre)
    # plot the true data and prediciton
    name = 'interval_'+str(i*4800+1)+'_iteration_num = '+str(iter_num)
#%%
    compare_plot(p_data_y,
                 fn_pre,
                 name_idx=name,
                 loss_name=loss_f,
                 loss_val=tt_loss_)
    t2 = time.time()
    t = t2-t1
    print('The training time is %f'%t)
#%% train and predict for splited data
#for i in np.arange(22,23):
#    split = pd.read_pickle('data/split/train_'+str(i+1)+'.pkl')
#    p_data_y = split.iloc[:,5].reshape((-1,1))  # test with all splited data y
#    fn_pre = np.load('data/pred/pred_'+str(i+1)+'.npy')
#    tt_loss_ = np.load('data/loss/loss_'+str(i+1)+'.npy')
#    loss_f = 'Bisquare'
#    name = 'interval_'+str(i*4800+1)+'_iteration_num = '+str(i+1)
#    compare_plot(p_data_y,
#                 fn_pre,
#                 name_idx=name,
#                 loss_name=loss_f,
#                 loss_val=tt_loss_)
#
#    print(i)
