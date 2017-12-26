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
faman[uct_idx]=-100
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
#%%
all_data = preprocessing.scale(all_data)
all_data_scaled = pd.DataFrame(preprocessing.scale(all_data_df),
                               columns = ['off','rx','ry','sx','sy','faaut','faman'])
#%%

#%%
inputs = all_data[:,0:5]
outputs = all_data[:,5].reshape((-1,1))
#%% plot 
from matplotlib.ticker import FuncFormatter
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(y*1./1659353*100)

    # The percent symbol needs escaping in latex
    if mpl.rcParams['text.usetex'] is True:
        return float(s)
    else:
        return format(float(s), '0.3f') +'%'
#------------------------------------------------------------------------------
plt.figure(figsize=(20,10),num=1)
fig = plt.gcf()
fig.set_size_inches(15,15)
mpl.rcParams.update({'font.size': 8})
fig.patch.set_facecolor('white')
#%%------------------------------------------------------------------------------
def hist_vis(subtitle,idx_):
    plt.figure(num=1)
    ax = plt.subplot(3,3,idx_+1)
    ax.hist(all_data[:,idx])    
    ax.set_title(subtitle)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
title_list = ['off','rx','ry','sx','sy','faaut','faman']
for idx,item in enumerate(title_list):
    hist_vis(item,idx)
plt.tight_layout()
#%%------------------------------------------------------------------------------
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



#%%0
start = 0
sample_num = 10000
end = start + sample_num
title_list = ['off','rx','ry','faaut','sx','sy','faman']

def part_vis(subtitle,idx_,data,fn=3):
    plt.figure(num=fn,figsize=(18,15))
    ax = plt.subplot(3,3,idx_+1)
    ax.scatter(np.arange(end-start),data[subtitle][start:end],s=0.5)    
    ax.set_title(subtitle)
    ax.grid()
    


for idx,item in enumerate(title_list):
    part_vis(item,idx,all_data_df,fn=3)
#        fig = plt.gcf()
#        fig.patch.set_facecolor('white')
#    part_vis(item,idx,all_data_df,fn=4)

# sx vs sy
plt.figure(num=3)
ax =plt.subplot(3,3,8)
ax.scatter(all_data_df['sx'],all_data_df['sy'],s=0.01)
ax.set_xlabel('sx data')
ax.set_ylabel('sy data')
ax.grid()

# rx vs vy
plt.figure(num=3)
ax =plt.subplot(3,3,9)
ax.scatter(all_data_df['rx'],all_data_df['ry'],s=0.01)
ax.set_xlabel('rx data')
ax.set_ylabel('ry data')
ax.grid()
plt.tight_layout()
#%%
#import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib import animation
#
#fig, ax = plt.subplots()
#
#off_data = off[0:2000]
#x = np.arange(len(off_data))
#line, = ax.plot(x, off_data)
#
#
#def animate(i):
#    line.set_ydata(off[x+i])  # update the data
#    return line,
#
#
## Init only required for blitting to give a clean slate.
#def init():
#    line.set_ydata(off_data)
#    return line,
#
#
#ani = animation.FuncAnimation(fig=fig, func=animate, frames=2000, init_func=init,
#                              interval=1, blit=True)
#
#
#plt.show()
