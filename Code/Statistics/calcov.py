# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:54:24 2018

@author: brucelau
"""

import numpy as np
import matplotlib.pyplot as plt

data =  np.array([-0.39,0.12,0.94,1.67,1.76,
                  2.44,3.72,4.28,4.92,5.53,
                  0.06,0.48,1.01,1.68,1.80,
                  3.25,4.12,4.60,5.28,6.22])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(data,bins=16)
