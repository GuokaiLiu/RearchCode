# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:13:38 2018

@author: guokai_liu
"""

from keras import optimizers

#%% Define MLP model
# parameter setting
cells = 100
nb_epoch = 96
batch_size = 240
learning_rate = 0.01
opt = optimizers.Adamax()
loss = 'mae'

#%% set the prediction file name and decide whether to plot the results
prediction_python_file_name = 'pre_off_rec.npy'
prediction_matlab_file_name = 'nfac-pre-off-rec.mat'

plot_result = True  # set True if: plot the results else Flase
reset = False        # set True if: reset model weight during training on different data slice  else False
updateoff = True    # set True if: update off is needed