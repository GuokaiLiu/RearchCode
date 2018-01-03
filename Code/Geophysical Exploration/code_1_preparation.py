# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:14:26 2017

@author: brucelau
"""

import scipy.io as sio
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
data = pd.read_pickle('data/split/train_1.pkl')
pre = np.load('data/pred/pred_1.npy')
p_data_y =data.iloc[:,5]
sl = 2400
sz=  0.2
plt.scatter(np.arange(sl),data['faaut'].iloc[0:sl],c='r',s=5,edgecolors='r',label='faaut')
plt.scatter(np.arange(sl),data['faman'].iloc[0:sl],c='g',s=sz,edgecolors='g',label='faman')
plt.scatter(np.arange(sl),pre[0:sl],c='b',s=sz,edgecolors='b',label='predict')
plt.ylim((0,1))
plt.legend()
plt.grid()
plt.plot()
#%%
plt.scatter(np.arange(sl),p_data_y,c='r',s=0.5,edgecolors='r')