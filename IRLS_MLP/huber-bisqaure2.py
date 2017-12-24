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
a = 1.5
b = 1
x_p = np.linspace(-2,2,50).reshape((-1,1)).astype('float32')
y_p = a*x_p**2 + b + +np.random.normal(0,0.1,len(x_p)).reshape((-1,1)).astype('float32')
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

y_ = tf.add(x*tf.matmul(x, w), b)

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
                              k*1./6*tf.ones_like(y_),              # plan A
                              k*1./6*(1-(1-((y_-y)*1./k)**2)**3)    # plan B
                             ))/ (2*len(x_data))                    # sum and average
    else:
        print('Wrong Loss function')
        
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        k_=1*4.685
        for i in range(1000):
            _,w_,b_,loss_= sess.run([train_step,w,b,loss],feed_dict={x:x_data,y:y_data,k:k_})
            y_pre = sess.run([y_],feed_dict={x:x_data,y:y_data,k:k_})
            y_pre = np.array(y_pre).reshape(-1)
            mar_error =np.median(np.abs(y_pre-y_n.reshape(-1)))
            k_ = mar_error*4.685
            print('w_:%f, b_:%f,loss_:%f'%(w_,b_,loss_))
            print('w_:%f, b_:%f'%(w_,b_))
        y_p = sess.run([y_],feed_dict={x:x_data,y:y_data})
    return w_,b_,np.array(y_p).reshape((-1,1))

# train L2 with pure data
w_0,b_0,yp0 = train_process(x_n,y_n,loss_fun='L2')    
# train L1 with noisy data
w_1,b_1,yp1 = train_process(x_n,y_n,loss_fun='L1')
# train L2 with noisy data
w_2,b_2,yp2 = train_process(x_n,y_n,loss_fun='Bisquare') 
# train L2 with noisy data
w_3,b_3,yp3 = train_process(x_n,y_n,loss_fun='Huber') 

#%%
fig = plt.gcf()
fig.set_size_inches(12,5)
# SUBPLOT LEFT
# subplot-1-L2+Original Data
ax = plt.subplot(121)
ax.scatter(x_n,y_n,label='$The Ground Truth: y=%fx^2+%f$'%(1.5,1),s=5)
ax.set_title('noisy data $y=1.5x^2+1+Normal(0,0.1)+noise$')
# subplot-2-L2+Pure Data
ax = plt.subplot(121)
ax.plot(x_p,yp0,'r',label='L2-Noisy_Data: $y=%fx^2+%f$'%(w_0,b_0))
# subplot-3-L1+Noisy Data
ax = plt.subplot(121)
ax.plot(x_n,yp1,'g--',label='L1-Noisy_Data: $y=%fx^2+%f$'%(w_1,b_1))
# subplot-4-L2+Noisy Data
ax = plt.subplot(121)
ax.plot(x_n,yp2,'k',label='Bisquare-Noisy_Data: $y=%fx^2+%f$'%(w_2,b_2))
# plot legend and grid on
ax.legend(loc='upper left',fontsize='x-small')
ax.grid()

# SUBPLOT RIGHT
# subplot-1-L2+Original Data
ax = plt.subplot(122)
ax.scatter(x_n,y_n,label='$The Ground Truth: y=%fx^2+%f$'%(1.5,1),s=5)
ax.set_title('noisy data $y=1.5x^2+1+Normal(0,0.1)+noise$')
# subplot-3-L2+Noisy Data
ax = plt.subplot(122)
ax.plot(x_n,yp2,'r--',label='Bisquare-Noisy_Data: $y=%fx^2+%f$'%(w_2,b_2))
# subplot-4-L2+Npisy Data
ax = plt.subplot(122)
ax.plot(x_n,yp3,'k',label='Huber-Noisy_Data: $y=%fx^2+%f$'%(w_3,b_3))

# plot legend and grid on
ax.legend(loc='upper left',fontsize='x-small')
ax.grid()

# super title
fig.suptitle('L2 loss vs L1 loss vs Huber and Bi-square loss',fontsize=20)