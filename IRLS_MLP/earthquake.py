# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 20:14:26 2017

@author: brucelau
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
#%% data preparation
# load the matlab data
earth_data = sio.loadmat('head.mat')
faaut = np.array(earth_data['faaut']).reshape(-1).astype('int')
faman = np.array(earth_data['faman']).reshape(-1).astype('int')
off = np.array(earth_data['off']).reshape(-1).astype('int')
rx = np.array(earth_data['rx']).reshape(-1).astype('int')
ry = np.array(earth_data['ry']).reshape(-1).astype('int')
sx = np.array(earth_data['sx']).reshape(-1).astype('int')
sy = np.array(earth_data['sy']).reshape(-1).astype('int')

# seperate the uncertain and certain data index
uct_idx = np.where(faman<0)[0]         # uncertain index 
uct_pct = len(uct_idx)*1.0/len(faman)  # uncertain percentage
idx_all = np.arange(len(faman))       
cet_idx = np.setxor1d(idx_all,uct_idx) # certain index

# select data for supervised learning
s_faaut = faaut[cet_idx]
s_faman = faman[cet_idx]
s_off = off[cet_idx]
s_rx = rx[cet_idx]
s_ry = ry[cet_idx]
s_sx = sx[cet_idx]
s_sy = sy[cet_idx]

# calculate and plot the difference bewteen automation and hand operation
diff_width = int(np.sqrt(len(s_faaut))+1)
padding =  np.zeros(diff_width**2-len(s_faaut))
diff = s_faaut-s_faman
diff_value = np.concatenate((diff,padding),axis=0).reshape((diff_width,diff_width))

# plot the difference and save the result
plt.matshow(diff_value)
plt.colorbar()
plt.title('defference = s_faaut - s_faman')

# train inputs and outputs arrays
inputs = np.zeros((len(s_faaut),5))
inputs[:,0]=s_off
inputs[:,1]=s_rx
inputs[:,2]=s_ry
inputs[:,3]=s_sx
inputs[:,4]=s_sy
outputs = np.zeros((len(s_faaut),3))
outputs[:,0] = s_faaut
outputs[:,1] = s_faman
outputs[:,2] = diff
#%%
stats.describe(outputs[:,2])


