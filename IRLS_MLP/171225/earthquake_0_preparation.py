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
#
all_data = np.zeros((len(faaut),7))
all_data[:,0] = off
all_data[:,1] = rx
all_data[:,2] = ry
all_data[:,3] = sx
all_data[:,4] = sy
all_data[:,5] = faaut
all_data[:,6] = faman
all_data_df = pd.DataFrame(all_data,columns = ['off','rx','ry','sx','sy','faaut','faman'])
#
certain_data = all_data[cet_idx,:]
uncertain_data = all_data[uct_idx,:]
#

#%%
inputs = all_data[:,0:5]
outputs = all_data[:,5].reshape((-1,1))
#%%
from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(y*1./1659353)

    # The percent symbol needs escaping in latex
    if mpl.rcParams['text.usetex'] is True:
        return float(s)
    else:
        return format(float(s), '0.3f') +'%'
#------------------------------------------------------------------------------
plt.figure(figsize=(20,10),num=1)
fig = plt.gcf()
fig.set_size_inches(15,3)
mpl.rcParams.update({'font.size': 8})
fig.patch.set_facecolor('white')
# SUBPLOT LEFT
# subplot-1-faaut
ax = plt.subplot(131)
ax.hist(all_data[:,-1])
ax.set_title('faman')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
# subplot-2-faman
ax = plt.subplot(132)
ax.hist(all_data[:,-2])
ax.set_title('faaut')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
# subplot-3-fanman-faaut
ax = plt.subplot(133)
ax.hist(all_data[:,-1]-all_data[:,-2])
ax.set_title('faman-faaut')
formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
#------------------------------------------------------------------------------
# calculate and plot the difference bewteen automation and hand operation
diff_width = int(np.sqrt(len(cet_idx))+1)
padding =  np.zeros(diff_width**2-len(cet_idx))
diff = (certain_data[:,-1]-certain_data[:,-2])*(1)
diff_value = np.concatenate((diff,padding),axis=0).reshape((diff_width,diff_width))

# plot the difference and save the result
plt.figure(figsize=(20,10),num=2)
fig = plt.gcf()
fig.set_size_inches(8,6)
mpl.rcParams.update({'font.size': 12})
fig.patch.set_facecolor('white')
plt.matshow(diff_value, fignum=2)
plt.colorbar()
plt.title('difference = s_faaut - s_faman')




#%%
## select data for supervised learning
#s_faaut = faaut[cet_idx]
#s_faman = faman[cet_idx]
#s_off = off[cet_idx]
#s_rx = rx[cet_idx]
#s_ry = ry[cet_idx]
#s_sx = sx[cet_idx]
#s_sy = sy[cet_idx]
#
## calculate and plot the difference bewteen automation and hand operation
#diff_width = int(np.sqrt(len(s_faaut))+1)
#padding =  np.zeros(diff_width**2-len(s_faaut))
#diff = s_faaut-s_faman
#diff_value = np.concatenate((diff,padding),axis=0).reshape((diff_width,diff_width))
#
## plot the difference and save the result
#plt.matshow(diff_value)
#plt.colorbar()
#plt.title('difference = s_faaut - s_faman')
#
## train inputs and outputs arrays
#inputs = np.zeros((len(s_faaut),5))
#inputs[:,0]=s_off
#inputs[:,1]=s_rx
#inputs[:,2]=s_ry
#inputs[:,3]=s_sx
#inputs[:,4]=s_sy
#outputs = np.zeros((len(s_faaut),3))
#outputs[:,0] = s_faaut
#outputs[:,1] = s_faman
#outputs[:,2] = diff

#%%

