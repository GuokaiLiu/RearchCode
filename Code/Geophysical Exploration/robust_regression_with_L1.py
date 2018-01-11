# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:14:26 2017

@author: brucelau
"""

import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import os
import math
from utils.next import Nextbacth
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mpl.style.use('classic')

# init
L = 1659353
delta = 4800

# seperate the uncertain and certain data index
def cu_idx(data):
    L=  len(data)
    uct_idx = np.where(data['faaut']<0)[0] # uncertain index 
    uct_pct = len(uct_idx)*1.0/L  # uncertain percentage
    _idx_all = np.arange(L)       
    cet_idx = np.setxor1d(_idx_all,uct_idx) # certain index
    return cet_idx,uct_idx,uct_pct

# load prediction data and save it as .mat file
def SaveMat(sorted_data):  
    # trans float to int
    for name in ('off','rx','ry','sx','sy','faaut','faaut2','faman'):
        sorted_data[name] = sorted_data[name].astype(int)
    # save the prediction data and remember to resave it in matlab for memory saving 
    b_dict = {col_name : sorted_data[col_name].values for col_name in sorted_data.columns.values}
    scipy.io.savemat('test_split_pos_neg.mat', {'struct':b_dict})
    
def predata():
    # data preparation: load the matlab data, sort and scale the data
    # Step-1:load
    global all_data_sc,all_data_st,faaut_mean,faaut_std,off_mean,off_std,L,delta,sl
    sl = int(L/delta)+1
    earth_data = sio.loadmat('data//head_cor.mat')
    all_data = np.zeros((L,7))
    all_data[:,0] = np.array(earth_data['off']).reshape(-1).astype('int')
    all_data[:,1] = np.array(earth_data['rx']).reshape(-1).astype('int')
    all_data[:,2] = np.array(earth_data['ry']).reshape(-1).astype('int')
    all_data[:,3] = np.array(earth_data['sx']).reshape(-1).astype('int')
    all_data[:,4] = np.array(earth_data['sy']).reshape(-1).astype('int')
    all_data[:,5] = np.array(earth_data['faaut']).reshape(-1).astype('int')
    all_data[:,6] = np.array(earth_data['faman']).reshape(-1).astype('int')
    all_data_df = pd.DataFrame(all_data,columns = ['off','rx','ry','sx','sy','faaut','faman'])
    # Step-2:sort
    all_data_st = all_data_df.sort_values(['sx','sy'],ascending=[True,False])
    # Step-3:scale
    all_data_sc = pd.DataFrame(preprocessing.scale(all_data_st),
                               columns = ['off','rx','ry','sx','sy','faaut','faman'])
# all_data_sc = (all_data_st-all_data_st.mean())/all_data_st.std()
    faaut_mean = all_data_st.mean()['faaut']
    faaut_std  = all_data_st.std()['faaut']
    off_mean = all_data_st.mean()['off']
    off_std = all_data_st.std()['off']
    
def train_data(data_):
    idx_cet,idx_uct,uct_pct = cu_idx(data_)
    return data_.iloc[idx_cet].iloc[:,0:5].as_matrix(),data_.iloc[idx_cet].iloc[:,5].values.reshape((-1,1))


# seperate the uncertain and certain data index
def split_train(train_):
    neg_idx = np.where(train_['off']<scaled_off_zero)[0] # uncertain index 
    _idx_all = np.arange(len(train_))       
    pos_idx = np.setxor1d(_idx_all,neg_idx) # certain index
    return train_.iloc[pos_idx],train_.iloc[neg_idx],pos_idx,neg_idx

def plot_splited_off(pos_idx_,neg_idx_,train_pos_,train_neg_):
    # make index 
    #idx = pd.DataFrame(np.arange(len(train)),index = train.index.values)
    #s1 = idx.loc[pos_idx].values
    #s2 = idx.loc[neg_idx].values
    fig = plt.figure(figsize=(9,6))
    plt.rcParams['figure.facecolor'] = 'white'
    ax = fig.add_subplot(111)
    ax.scatter(pos_idx_,train_pos_['off'],edgecolors='r',s=2)
    ax.scatter(neg_idx_,train_neg_['off'],edgecolors='b',s=2)
    ax.grid()
def plottrain(x):
    plt.rcParams['figure.facecolor']='white'
    plt.ion()
    fig = plt.figure(figsize=(15,9))
    fig.suptitle('Data Visualizaiton',fontsize=24)
    sl = np.arange(len(x))
    for idx,item in enumerate(['off','rx','ry','sx','sy','faman','faaut']):
        ax = fig.add_subplot(3,3,idx+1)
        ax.scatter(sl,x.loc[:,item],s=1)
        ax.set_title(item)
        ax.grid()
    # 
    ax = fig.add_subplot(3,3,8)
    ax.scatter(x.loc[:,'rx'],x.loc[:,'ry'],s=1)
    ax.set_title('rx vs ry') 
    ax.set_xlabel('rx')
    ax.set_ylabel('ry')
    ax.grid()
    #
    ax = fig.add_subplot(3,3,9)
    ax.scatter(x.loc[:,'sx'],x.loc[:,'sy'],s=1)
    ax.set_title('sx vs sy')
    ax.set_xlabel('sx')
    ax.set_ylabel('sy')
    ax.grid()
    plt.savefig('plot/datavis.png')

#
plottrain(all_data_st[0:4800])

def plot_part_prediction(part_data,start,end,aut_name='faaut',pre_name='faaut2'):
    faaut = part_data.loc[start:end-1,aut_name]
    faaut2 = part_data.loc[start:end-1,pre_name]
    fig = plt.figure(figsize=(12,6))
    plt.ioff()
    plt.rcParams['figure.facecolor']='white'
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(end-start),faaut,label='faaut',edgecolors='r',s=1)
    ax.scatter(np.arange(end-start),faaut2,label='faaut-pre',edgecolors='b',s=1)
    ax.legend()
    ax.grid()
    ax.set_xlabel('number index')
    ax.set_ylabel('faaut vs faaut-pre')
    ax.set_title('Robust Regression with L1 loss')
    ax.set_ylim(0,6000)
    plt.tight_out()
    plt.savefig('plot/'+str(start)+'.png')
    
def plot_final_result(data,c1='faaut',c2='faaut2'):
    s = 0
    e = s + delta
    for i in range(346):
        if e>=L:
            e=L
        plot_part_prediction(data,s,e)
        s += delta
        e += delta
#def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
#    # add one more layer and return the output of this layer
#    layer_name = 'layer%s' % n_layer
#    with tf.name_scope(layer_name):
#        with tf.name_scope('weights'):
#            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
#            tf.summary.histogram(layer_name + '/weights', Weights)
#        with tf.name_scope('biases'):
#            biases = tf.Variable(tf.zeros([1, out_size]), name='b')
#            tf.summary.histogram(layer_name + '/biases', biases)
#        with tf.name_scope('Wx_plus_b'):
#            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
#        if activation_function is None:
#            outputs = Wx_plus_b
#        else:
#            outputs = activation_function(Wx_plus_b, )
#        tf.summary.histogram(layer_name + '/outputs', outputs)
#    return outputs
        
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

        
def train_process(train_x,train_y,px,py):
    with tf.Session() as sess:
        sn = Nextbacth(np.arange(len(train_x)))
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ls = []
        
        for i in range(2000):
            # update learning rate
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decaprespeed)
            
            # update batch data
            label = sn.next_batch(batch_size)
            batch_x = train_x[label]
            batch_y = train_y[label]
            
            # train model
            sess.run(train_step, feed_dict={xs: batch_x, 
                                            ys: batch_y,
                                            lr:learning_rate})
            # test loss
            if i % 50 == 0:
                loss_,pre = sess.run([loss,prediction],feed_dict={xs: px, 
                                                 ys: py,
                                                 lr: learning_rate})
                ls.append(loss_)
#                print(loss_)
    return pre

    
#%%
#SaveMat(faaut_std,faaut_mean,all_data_st)
#plot_splited_off(pos_idx,neg_idx,train_pos,train_neg)
#%%
batch_size = 240
max_learning_rate = 0.2
min_learning_rate = 0.001
decaprespeed = 500.0
#
input_dim = 5
out1 = in2 = 100 
out2 = in3 = 50
out3 = in4 = 25
output_dim = 1

# Make up some real data
#x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, input_dim], name='x_input')
    ys = tf.placeholder(tf.float32, [None, output_dim], name='y_input')
    lr = tf.placeholder(tf.float32,name = 'my_lr')

# add hidden layer-1
l1 = add_layer(xs, input_dim, out1, n_layer=1, activation_function=tf.nn.relu)
# add hidden layer-2
l2 = add_layer(l1, in2, out2, n_layer=2, activation_function=tf.nn.relu)
# add hidden layer-3
l3 = add_layer(l2, in3, out3, n_layer=3, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l3, in4, 1, n_layer=4, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(ys - prediction),
#                                        reduction_indices=[1]))
    loss = tf.reduce_sum(tf.abs(prediction-ys)) / (2*batch_size)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir logs
#%%
predata()
scaled_off_zero = (0-off_mean)/off_std
start_p = 0
end_p = start_p+delta
all_data_sc['faaut2']=np.zeros(len(all_data_sc))

for i in range(346):
    t1 = time.time()
    print(start_p,end_p)
    if end_p>=L:
        end_p = L
    # select sequence data
    data = all_data_sc[start_p:end_p]
    # prepare data for training and prediction
    data_pos, data_neg, pos_idx, neg_idx = split_train(data)
    # for train 
    train_pos_rx, train_pos_ry = train_data(data_pos)
    train_neg_rx, train_neg_ry = train_data(data_neg)
    # for predciton
    
    p_pos_x = data_pos.iloc[:,0:5].as_matrix()
    p_pos_y = data_pos.iloc[:,5].values.reshape((-1,1))
    p_neg_x = data_neg.iloc[:,0:5].as_matrix()
    p_neg_y = data_neg.iloc[:,5].values.reshape((-1,1))
    # train and prediction
    faaut2_pos = train_process(train_pos_rx,train_pos_ry,p_pos_x,p_pos_y)
    faaut2_neg = train_process(train_neg_rx,train_neg_ry,p_neg_x,p_neg_y)
    # append prediction results
#    data_pos['faaut2']=pd.Series(faaut2_pos.reshape(-1),index=data_pos.index)
#    data_neg['faaut2']=pd.Series(faaut2_neg.reshape(-1),index=data_neg.index)
    # appending the predction result to the origina scaled dataframe
    all_data_sc.loc[list(data_pos.index),'faaut2']=faaut2_pos.reshape(-1)
    all_data_sc.loc[list(data_neg.index),'faaut2']=faaut2_neg.reshape(-1)
    start_p += delta
    end_p += delta
    t2 = time.time()
    t = t2-t1
    print('One Round-%d cost time: %s seconds\n'%(i,t))
#%%
#plt.scatter(np.arange(len(p_neg_y)),p_neg_y,edgecolors='b',s=1)
#plt.scatter(np.arange(len(faaut2_neg)),faaut2_neg,edgecolors='r',s=1)
#plt.scatter(np.arange(len(p_pos_y)),p_pos_y,edgecolors='b',s=1)
#plt.scatter(np.arange(len(faaut2_pos)),faaut2_pos,edgecolors='r',s=1)
plot_part_prediction(all_data_sc,start_p-delta,end_p-delta,aut_name='faaut',pre_name='faaut2')
#%%
all_data_st['faaut2']=np.zeros(L)
all_data_st['faaut2']=all_data_sc['faaut2'].as_matrix()*faaut_std+faaut_mean
result = all_data_st.sort_index()
#%%
SaveMat(result)
#%%
plot_final_result(result)