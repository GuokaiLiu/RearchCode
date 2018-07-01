# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:19:48 2018

@author: guokai_liu
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import optimizers
import numpy as np
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pylab as plot
import keras.backend as K
import time
import scipy
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#%% define functions
# define data visualizaiton function
def data_vis(df_data):
    x = np.arange(len(df_data))
    fig = plt.figure(figsize=(12,6))
    # univariate
    for idx,item in enumerate(df_data.columns[0:-1]):
        ax = fig.add_subplot(3,3,idx+1)
        ax.scatter(x,df_data[item],s=0.1,c='r')
        ax.set_title(item)
    # multivariate
    ax = fig.add_subplot(3,3,8)
    ax.scatter(df_data['rx'],df_data['ry'],s=0.05,c='r')
    ax.set_xlabel('rx')
    ax.set_ylabel('ry')
    ax = fig.add_subplot(3,3,9)
    ax.scatter(df_data['sx'],df_data['sy'],s=10,c='r')
    ax.set_xlabel('sx')
    ax.set_ylabel('sy')
    plt.tight_layout()

def compare(d_p,d_y,title,s,cut,train=False):
    if train==True:
        label2 = test
    else:
        label2 = 'faman'
    plt.ioff()
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    x = np.arange(len(d_p))
    ax.scatter(x,d_p,s=0.1,c='r',label='pred')
    ax.scatter(x,d_y,s=0.1,c='b',label=label2)
    if cut==True:
        ax.set_ylim(-2,)
        pass
    plt.legend(loc='upper right')
    plt.grid()
    plt.title(title,fontsize=20)
    plt.tight_layout()
    plt.savefig('pre-pic3/nfac-%d.png'%s,dpi=600)
    
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            
def slice_data(start,delta):
    s = start
    e = s+delta if s+delta<=l else l
    print('Section:[%d, %d]'%(s,e))
    df_sel = df.iloc[s:e]
    df_fil = df_sel[df_sel['nc']!=-9999]
    df2arr_sel = df_sel.as_matrix()
    df2arr_fil = df_fil.as_matrix()
    scaler_train_input = StandardScaler().fit(df2arr_fil)
    normalized_train = scaler_train_input.transform(df2arr_fil)
    normalized_test = scaler_train_input.transform(df2arr_sel)
    # training and testing data
    X_train = normalized_train[:,0:5]
    X_test = normalized_test[:,0:5]
    
    # preprocessing labels
    # the origina selected data
    nfac_test  = df2arr_sel[:,7].reshape(-1,1)
    faman_test  = df2arr_sel[:,6].reshape(-1,1)
    
    # the original filted data
    nfac_train  = df2arr_fil[:,7].reshape(-1,1)
    
    # calculate training mean and var
    # training and testing data with nfac mean adnd var
    scaler_nfac  = StandardScaler().fit(nfac_train)
    y_train_nc = scaler_nfac.transform(nfac_train)
    y_test_nc = scaler_nfac.transform(nfac_test)
    y_test_ncfm = scaler_nfac.transform(faman_test)
    
    return X_train, X_test,y_train_nc,y_test_nc,y_test_ncfm,scaler_nfac,faman_test


def update_off(rx_,ry_,sx_,sy_,of_):
    # np.cos(np.arctan2(3,4)) = 0.8
    dxx = rx - sx
    dyy = ry - sy
    dtof= np.cos(np.arctan2(dxx,dyy))
    print(dtof.shape,of_.shape)
    return of_*np.sign(dtof)

def load_pred(pre_data_path):
    pre1 = [i for i in np.load(pre_data_path)]
    pre2 = []
    for i in pre1:
        for j in i:
            pre2.append(j)
    pre3 = np.array(pre2)
    print(len(pre3))
    df['pre']=pre3

def SaveMat(numpy_data,save_name):
    # transfer the data type if needed
    numpy_data['pre'] = numpy_data['pre']#.astype(int)
    # save the prediction data and remember to resave it in matlab for memory saving 
    b_dict = {col_name : numpy_data[col_name].values for col_name in numpy_data.columns.values}
    scipy.io.savemat(save_name, {'struct':b_dict})
    
def vis_inline_dis():
    idx = mat['inline'].reshape((-1))
    x = np.arange(len(idx))
    plt.figure(figsize=(12,9))
    plt.scatter(x,idx,s=0.1)
    plt.title('Inline distribution')
    plt.savefig('Inline distribution.jpg',dpi=600)
    
    
def vis_inline_bar():
    inline_idx = mat['inline'].reshape((-1))
    inline_pds = pd.DataFrame()
    inline_pds['inline'] = inline_idx
    grouped = inline_pds['inline'].groupby(inline_pds['inline'])
    s = grouped.count()
    i = s.index
    bar_data = pd.DataFrame()
    bar_data['index']=i
    bar_data['count']=s.values
    plt.bar(range(len(bar_data)),bar_data['count'])
    print(bar_data.describe())
    
    
def slice_shot_zone():
    # cunsum the 'inline' data for group in .mat file
    inline_idx = mat['inline'].reshape((-1))
    inline_ser = pd.Series(inline_idx)
    grouped = inline_ser.groupby(inline_ser.values,sort=False)
    s = grouped.count()
    cs = s.cumsum()
    
    # caluculate the slice zone: start → end
    s = 0
    slice_start = [0]
    slice_end = []
    for idx,item in enumerate(cs.values):
        if idx!=0 and idx%3==0:
            slice_end.append(cs.iloc[idx-1])
            slice_start.append(cs.iloc[idx-1]+1)
    slice_end.append(cs.iloc[-1])
    
    # calcutae the zone length
    slice_zone = pd.DataFrame()
    slice_zone['start'] = slice_start
    slice_zone['end'] = slice_end
    slice_zone['delta'] = [slice_end[i]-slice_start[i] for i in range(len(slice_start))]
    return slice_zone
#%% load data
# data preparation
mat = sio.loadmat('picktime_test_resort_header3.mat')
keys_1 = ['rx','ry','sx','sy','off','faman','faaut','nfac']
keys_2 = ['nfac','inline']  
rx= np.array(mat['rx']).reshape((-1))
ry= np.array(mat['ry']).reshape((-1))
sx= np.array(mat['sx']).reshape((-1))
sy= np.array(mat['sy']).reshape((-1))
of= np.array(mat['off']).reshape((-1))
fa= np.array(mat['faaut']).reshape((-1))
fm= np.array(mat['faman']).reshape((-1))
nc= np.array(mat['nfac']).reshape((-1))

# preparation
df = pd.DataFrame()
df['rx'] = rx
df['ry'] = ry
df['sx'] = sx
df['sy'] = sy
df['of'] = of
df['fa'] = fa
df['fm'] = fm
df['nc'] = nc*2
df2arr = df.as_matrix()
#%% slice data for training and testing
# data filer
l = len(df)
# Make list to append the predictions
pre = []
te = []
#%% Define the model
cells = 100
model = Sequential()
model.add(Dense(cells,activation='relu',input_shape=(5,)))
model.add(Dense(cells//2,activation='relu'))
model.add(Dense(cells//4,activation='relu'))
model.add(Dense(1))
model.summary()
#%%
nb_epoch = 96
learning_rate = 0.01
decay_rate = learning_rate / nb_epoch
momentum = 0.9
#sgd = optimizers.SGD(lr=0.01,decay=1e-6)
#opt = SGD(lr=learning_rate, decay=learning_rate/nb_epoch, momentum=momentum, nesterov=False)
opt = optimizers.Adamax()
model.compile(loss='mae',optimizer=opt,metrics=['acc'])


#%% Set the training labels
test = 'nfac'
t1 = time.time()

shot_zone = slice_shot_zone()

#%%
#for i in range(int(l/4800)+1):
#for i in range(3):

for i in range(len(shot_zone)):
    p = shot_zone.iloc[i]['start']
    d = shot_zone.iloc[i]['delta']
#    print('Start %d, End %d'%(p,d))

#    p = i*4800
    X_train, X_test,y_train_nc,y_test_nc,y_test_ncfm,scaler_nfac,faman_test = slice_data(p,d)
    y_train_label = y_train_nc
    y_test_label = y_test_ncfm
#    te.append(X_test)
    
    loss_before = np.mean(np.abs(model.predict(X_train)-y_train_label))
    
    histrory = model.fit(X_train,y_train_label,
                         batch_size=240,
                         epochs = nb_epoch,
                         verbose= 0)
    #%% inverse prediciton to the same scale as labels 
    y_pred_test  = model.predict(X_test).reshape(-1)
    inv_pre_nc = scaler_nfac.inverse_transform(y_pred_test)
    pre.append(inv_pre_nc)
    compare(inv_pre_nc,faman_test,  '在原始数据'+test+'(size=%d)上进行测试预测的结果'%d,p,cut=True)
    #%% Test if reset_weights function works
    #loss_after = np.mean(np.abs(model.predict(X_train)-y_train_label))
    #print('loss_before:\n',loss_before)
    #print('loss_after:\n',loss_after)
    #%% reset model
    reset_weights(model)
    t2 = time.time()
    print('Time Cost is: ',(t2 - t1)/60, 'mins')
np.save('pre_off_rec.npy',pre)
#%%
pre_data_path = 'pre_off_test2.npy'
load_pred(pre_data_path)
#%%
SaveMat(df,'nfac-pre-off-rec3.mat')


#%%
np.save('pre_off_test2.npy',pre)
pre_data_path = 'pre_off_test2.npy'
data = np.load(pre_data_path)


#%%
c = 0
for i in range(len(pre)):
    c += (len(pre[i]))
        

    
    
    
    
