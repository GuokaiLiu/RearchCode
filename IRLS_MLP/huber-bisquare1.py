# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 13:50:57 2517

@author: brucelau
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%% make data and plot
# make data
x_p = np.linspace(-5,5,50).reshape((-1,1)).astype('float32')
y_p = 0.6*x_p + 5 + +np.random.normal(0,0.1,len(x_p)).reshape((-1,1)).astype('float32')
y_p = y_p.reshape((-1,1)).astype('float32')
x_n = x_p.copy()
y_n = y_p.copy()
y_n[40]=15
y_n[35]=12
y_n[30]=8
y_n[25]=5

#%% build the model 
x = tf.placeholder(tf.float32,[None,1],name='inputs')
y = tf.placeholder(tf.float32,[None,1],name='outputs')
w = tf.Variable(tf.ones([1,1]))
b = tf.Variable(tf.ones([1,1]))
k = tf.placeholder(tf.float32,name ='tunning_constant')

y_ = tf.add(tf.matmul(x, w), b)

def train_process(x_data,y_data,loss_fun='L2'):
    if loss_fun=='L2':
        loss = tf.reduce_sum(tf.pow(y_-y,2)) / (2*len(x_data))
    elif loss_fun=='L1':
        loss = tf.reduce_sum(tf.abs(y_-y)) / (2*len(x_data))
    elif loss_fun=='Huber':
        loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(y_-y),k),   # condition
                              (k*tf.abs(y_-y)-0.5*k**2)/len(x_data),       # plan A
                              0.5*tf.pow(y_-y,2)                    # plan B
                             ))/ (2*len(x_data))                    # sum and average
    elif loss_fun=='Bisquare':
        loss = tf.reduce_sum(tf.where(tf.greater(tf.abs(y_-y),k),   # condition
                              k*1./6*tf.ones_like(y_),                # plan A
                              k*1./6*(1-(1-((y_-y)*1./k)**2)**3)             # plan B
                             ))/ (2*len(x_data))                    # sum and average
    
    else:
        print('Wrong Loss function')
        
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        k_=1*4.685
        for i in range(5000):
            _,w_,b_,loss_= sess.run([train_step,w,b,loss],feed_dict={x:x_data,y:y_data,k:k_})
            y_pre = sess.run([y_],feed_dict={x:x_data,y:y_data,k:k_})
            y_pre = np.array(y_pre).reshape(-1)
            mar_error =np.median(np.abs(y_pre-y_n.reshape(-1)))
            k_ = mar_error*4.685
            print('w_:%f, b_:%f,loss_:%f'%(w_,b_,loss_))
    return w_,b_,y_pre

# train L2 with pure data
w_0,b_0,y_pre0 = train_process(x_p,y_p,loss_fun='L2')    
# train L1 with noisy data
w_1,b_1,y_pre1 = train_process(x_n,y_n,loss_fun='L1')
# train L2 with noisy data
w_2,b_2,y_pre2 = train_process(x_n,y_n,loss_fun='L2') 
# train Huber with noisy data
w_3,b_3,y_pre3 = train_process(x_n,y_n,loss_fun='Bisquare') 

yp0 = x_p*w_0+b_0
yp1 = x_n*w_1+b_1
yp2 = x_n*w_2+b_2
yp3 = x_n*w_3+b_3
#%%
fig = plt.gcf()
fig.set_size_inches(12,5)
# LEFT
# subplot-1-L2+Original Data
ax = plt.subplot(121)
ax.scatter(x_n,y_n,label='$The Ground Truth: y=%f+%f$'%(0.6,5),s=5)
ax.set_title('noisy data $y=0.6x+5+Normal(0,0.1)+noise$')
# subplot-2-L2+Pure Data
ax = plt.subplot(121)
ax.plot(x_p,yp0,'r',label='L2-Pure_Data: $y=%f+%f$'%(w_0,b_0))
# subplot-3-L1+Noisy Data
ax = plt.subplot(121)
ax.plot(x_n,yp1,'g--',label='L1-Noisy_Data: $y=%f+%f$'%(w_1,b_1))
# subplot-4-L2+Npisy Data
ax = plt.subplot(121)
ax.plot(x_n,yp2,'b',label='L2-Noisy_Data: $y=%f+%f$'%(w_2,b_2))
ax.legend(loc='upper left',fontsize='x-small')
ax.grid()

# RIGHT
# subplot-1-L2+Original Data
ax = plt.subplot(122)
ax.scatter(x_n,y_n,label='$The Ground Truth: y=%f+%f$'%(0.6,5),s=5)
ax.set_title('noisy data $y=0.6x+5+Normal(0,0.1)+noise$')
# subplot-2-L2+Pure Data
ax = plt.subplot(122)
ax.plot(x_p,yp0,'r',label='L2-Pure_Data: $y=%f+%f$'%(w_0,b_0))
# subplot-3-L1+Noisy Data
ax = plt.subplot(122)
ax.plot(x_n,yp3,'g--',label='Huber-Noisy_Data: $y=%f+%f$'%(w_3,b_3))
# subplot-4-L2+Npisy Data
ax = plt.subplot(122)
ax.plot(x_n,yp2,'b',label='L2-Noisy_Data: $y=%f+%f$'%(w_2,b_2))
ax.legend(loc='upper left',fontsize='x-small')
ax.grid()