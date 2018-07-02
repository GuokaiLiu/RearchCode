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
import numpy as np
from keras.layers import Dense
import keras.backend as K
import time
import scipy
from para_card import cells, nb_epoch, batch_size, opt, loss
from para_card import prediction_python_file_name,prediction_matlab_file_name, plot_result, reset, updateoff

params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams['font.sans-serif']=['SimHei'] # for chinese
plt.rcParams['axes.unicode_minus']=False   # for '-'
#%% Define functions

def data_vis(df_data):
    """
    Function: A data visualizaiton function for mat file
    ------
    Input: Pandas DataFrame data
    Output:a visualization figure
    """
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


def compare(d_p,d_y,title,s,d,clip=True,train=False):
    """
    Function: Plot the prediction results with faman data
    ------
    Input: 
        d_p: prediction data
        d_y: faman data
        s: size for line width
        clip: if True, the limitation for y axis will be set to -2
    Output: 
        A figure of the results
    """
    if train==True:
        label2 = test_data
    else:
        label2 = 'faman'
    plt.ioff()
    fig = plt.figure(figsize=(9,5))
    ax = fig.add_subplot(111)
    x = np.arange(len(d_p))
    ax.scatter(x,d_p,s=0.1,c='r',label='pred')
    ax.scatter(x,d_y,s=0.1,c='b',label=label2)
    if clip==True:
        ax.set_ylim(-2,)
        pass
    plt.legend(loc='upper right')
    plt.grid()
    plt.title(title,fontsize=20)
    plt.tight_layout()
    plt.savefig('results/nfac-start-%d-delta-%d.png'%(s,d),dpi=600)
    
def update_off(rx_,ry_,sx_,sy_,of_):
    """
    Function: update off values
    ------
    Input:
        raw rx, ry, sx, sy, off
    Output:
        updated rx, ry, sx, sy, off
    """
    # np.cos(np.arctan2(3,4)) = 0.8
    dxx = rx - sx
    dyy = ry - sy
    dtof= np.cos(np.arctan2(dxx,dyy))
    print(dtof.shape,of_.shape)
    return of_*np.sign(dtof)
    
def reset_weights(model):
    """
    Function: reset the input model
    ------
    Input: a trained model
    Output: a reset model
    """
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            
def slice_shot_zone():
    """
    Function: Slicing raw data according to the inline column.
    Return: A DataFrame which contains the start points, end points and delta length
    """
    inline_idx = mat['inline'].reshape((-1))
    # make start points list for slice zone
    start_idx =  0
    list_start = []
    list_start.append(start_idx)\
    
    for idx,s in enumerate(inline_idx):
        if s!=inline_idx[start_idx]:
            start_idx = idx
            list_start.append(start_idx)
    
    # make end points list for slice zone
    list_end = []
    for s in list_start[1:]:
        list_end.append(s-1)
    list_end.append(len(inline_idx)-1)
    
    # make slice zone list
    list_delta =[j-i+1 for (j,i) in zip(list_end,list_start)]
            
     
    # make slice zone 
    slice_zone = pd.DataFrame()
    slice_zone['start'] = list_start
    slice_zone['end'] = list_end
    slice_zone['delta'] = list_delta
    
    # judge [the length of slice_zone]==[the length of all data] 
    
    assert sum(slice_zone['delta'])==len(inline_idx)
    return slice_zone            
            
            
def slice_data(start,delta):
    """
    Function: Slice data for training and testing 
    Input: 
        start: start point for slicing
        delta: slicing length 
    Output:
        X_train: filted and standardized input data  which exclude -9999 nfac values
        X_test: standardized input data which include -9999 nfac data with the same mean and var as X_train
        y_train_nc: filted and standardized label data  which exclude -9999 nfac values
        y_test_nc: standardized nfac label 
        y_test_ncfm: standardized faman label
        scaler_ncfac: scalar for inverse_transform
        faman_test: faman data for testing 
            
    StandardScaler Ref:
        [1] http://scikit-learn.org/stable/modules/preprocessing.html
        [2] https://blog.csdn.net/qq_33039859/article/details/80191863
    
    """
    s = start
    e = s+delta if s+delta<=l else l
    print('Section:[%d, %d]'%(s,e))
    
    # slice data and select 'nfac' values that are not equal to -9999
    df_sel = df.iloc[s:e]
    df_fil = df_sel[df_sel['nc']!=-9999]
    
    # set dataframe as matrix
    df2arr_sel = df_sel.values
    df2arr_fil = df_fil.values
    
    # apply sklearn StandarScaler() to preprocess the inputs for each columns(rx, ry, sx, sy, off)
    # using the mean and var from filted nfac data excluding -9999 values
    scaler_train_input = StandardScaler().fit(df2arr_fil)
    normalized_train = scaler_train_input.transform(df2arr_fil)
    normalized_test = scaler_train_input.transform(df2arr_sel)
    
    # training and testing data (zero mean and unit variance)
    X_train = normalized_train[:,0:5]
    X_test = normalized_test[:,0:5]
    
    ## preprocessing labels
    # the original filted data for training
    nfac_train  = df2arr_fil[:,7].reshape(-1,1)
    
    # the origina selected data for testing
    nfac_test  = df2arr_sel[:,7].reshape(-1,1)
    faman_test  = df2arr_sel[:,6].reshape(-1,1)  

    ## standardize training and testing labels    
    # calculate the nfac_train mean and var through StandardScaler().fit()
    scaler_nfac  = StandardScaler().fit(nfac_train)
    
    ## apply nfac_train mean and var to [nfac_train, nfac_test, faman_test] through .transform()
    # use nfac_train mean and var to standardize nfac training data
    y_train_nc = scaler_nfac.transform(nfac_train)
    # use nfac_train mean and var to standardize nfac testing data
    y_test_nc = scaler_nfac.transform(nfac_test)
    # use nfac_train mean and var to standardize faman testing data
    y_test_ncfm = scaler_nfac.transform(faman_test)
    
    return X_train, X_test,y_train_nc,y_test_nc,y_test_ncfm,scaler_nfac,faman_test




def load_pred(pre_data_path):
    """
    Function: Load the prediction result
    Input: 
        pre_data_path: predcition data path
    Output:
        Add a new column named 'pre'to df which contains the raw .mat file 
    """
    
    pre1 = [i for i in np.load(pre_data_path)]
    pre2 = []
    for i in pre1:
        for j in i:
            pre2.append(j)
    pre3 = np.array(pre2)
    print(len(pre3))
    df['pre']=pre3

def savemat(numpy_data,save_name):
    """
    Function: save python data to  .mat file
    ------
    Input:
        numpy_data: numpy data
    Output: 
        .mat file named 'save_name.mat'
    """
    # transfer the data type if needed
    numpy_data['pre'] = numpy_data['pre']#.astype(int)
    # save the prediction data and remember to resave it in matlab for memory saving 
    b_dict = {col_name : numpy_data[col_name].values for col_name in numpy_data.columns.values}
    scipy.io.savemat(save_name, {'struct':b_dict})
#%% run the main code    
if __name__ == '__main__':
    #%% Data preparation
    # load data
    mat = sio.loadmat('picktime_test_resort_header3.mat')
    keys_1 = ['rx','ry','sx','sy','off','faman','faaut','nfac']
    keys_2 = ['nfac','inline']  
    rx= np.array(mat['rx']).reshape((-1))
    ry= np.array(mat['ry']).reshape((-1))
    sx= np.array(mat['sx']).reshape((-1))
    sy= np.array(mat['sy']).reshape((-1))
    of= np.array(mat['off']).reshape((-1))
    if updateoff==True:
        of = update_off(rx,ry,sx,sy,of)
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
    
    # set data as matix
    l = len(df)
    df2arr = df.values

    # MLP model and compile
    model = Sequential()
    model.add(Dense(cells,activation='relu',input_shape=(5,)))
    model.add(Dense(cells//2,activation='relu'))
    model.add(Dense(cells//4,activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss=loss,optimizer=opt,metrics=['acc'])
    
    #%% Recording preparation
    # make list to append the predictions
    pre = []

    
    #%% Training and Testing
    # preparation
    test_data = 'nfac'
    t1 = time.time()
    shot_zone = slice_shot_zone()
    
    for i in range(len(shot_zone)):
        # prepare the slicing data
        p = shot_zone.iloc[i]['start']
        d = shot_zone.iloc[i]['delta']
        X_train, X_test,y_train_nc,y_test_nc,y_test_ncfm,scaler_nfac,faman_test = slice_data(p,d)
        y_train_label = y_train_nc
        y_test_label = y_test_ncfm
        # model fitting
        histrory = model.fit(X_train,y_train_label,
                             batch_size=batch_size,
                             epochs = nb_epoch,
                             verbose= 0)
        # inverse prediciton to the same scale as labels 
        y_pred_test  = model.predict(X_test).reshape(-1)
        inv_pre_nc = scaler_nfac.inverse_transform(y_pred_test)
        pre.append(inv_pre_nc)
        
        # plot result
        if plot_result == True:
            compare(inv_pre_nc,faman_test,  '在原始数据'+test_data+'(size=%d)上进行测试预测的结果'%d,p,d,clip=True)
    
        # reset model
        if reset == True:
            reset_weights(model)

        t2 = time.time()
        print('Time Cost is: ',(t2 - t1)/60, 'mins')
    np.save(prediction_python_file_name,pre)
    #%% Load the prediciton file and save it as ,mat file
    pre_data_path = prediction_python_file_name
    load_pred(pre_data_path)
    savemat(df,prediction_matlab_file_name)



        

    
    
    
    
