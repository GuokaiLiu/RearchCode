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
mpl.style.use('classic')
#%%
# seperate the uncertain and certain data index
def cu_idx(data):
    L=  len(data)
    uct_idx = np.where(data['faaut']<0)[0]         # uncertain index 
    uct_pct = len(uct_idx)*1.0/L  # uncertain percentage
    _idx_all = np.arange(L)       
    cet_idx = np.setxor1d(_idx_all,uct_idx) # certain index
    return cet_idx,uct_idx,uct_pct

#%% data preparation
# load the matlab data
L = 1659353
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
all_data_st = all_data_df.sort_values(['sx','sy'],ascending=[True,False])
all_data_sc = pd.DataFrame(preprocessing.scale(all_data_st),columns = ['off','rx','ry','sx','sy','faaut','faman'])
all_data_sc.to_pickle('data/all_data_sc.pkl')

#%%
faaut_mean = all_data_st.mean()['faaut']
faaut_std  = all_data_st.std()['faaut']
scaled_my = (all_data_df['faaut']-faaut_mean)*1.0/faaut_std
scaled_pd = preprocessing.scale(all_data_df['faaut'])
#%% split scaled data
scaled_data = pd.read_pickle('data/all_data_sc.pkl')
delta = 4800
sl = int(len(scaled_data)/delta)+1

i=0
start = i*delta
end = (i+1)*delta
while(end<len(scaled_data)):
    scaled_data.iloc[start:end].to_pickle('data/split/train_'+str(i+1)+'.pkl')
    start = start+delta
    end  = end+delta
    print(end)
    i = i + 1
scaled_data.iloc[start:end].to_pickle('data/split/train_'+str(i+1)+'.pkl')

# calculate the error ratio    
e = []
for i in range(sl):
    split = pd.read_pickle('data/split/train_'+str(i+1)+'.pkl')
    idx_cet,idx_uct,uct_pct = cu_idx(split)
    train = split.iloc[idx_cet]
    e.append(uct_pct)

#%%
    #%% get the invsese scaled prediction
# laod the scaled predictions 
pre1 = np.load('data/pred/pred_'+str(1)+'.npy')
for idx in np.arange(1,346):
    tmp = np.load('data/pred/pred_'+str(idx+1)+'.npy')
    pre1 = np.concatenate((pre1,tmp),axis=0)

# inverse scaling 
pre2 = pre1*faaut_std+faaut_mean
# load the local errors
all_data_pr = all_data_st
all_data_pr['faaut2']=pre2
#%%
for name in ('off','rx','ry','sx','sy','faaut','faaut2','faman'):
    all_data_pr[name] = all_data_pr[name].astype(int)
#%% save the prediction data and remember to resave it in matlab for memory saving 
a_dict = {col_name : all_data_pr[col_name].values for col_name in all_data_pr.columns.values}
scipy.io.savemat('head_cor_pre2.mat', {'struct':a_dict})
#%%
