# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:34:00 2017

@author: brucelau
"""

import numpy as np


    
loss = []
for idx in np.arange(346):
    tmp = np.load('data/loss/loss_'+str(idx+1)+'.npy')
    loss.append(float(tmp))