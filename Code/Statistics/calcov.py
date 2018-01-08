# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:54:24 2018

@author: brucelau
"""

import numpy as np
import matplotlib.pyplot as plt

def conv(data):
    '''
    u = E(X)
    v = E(Y)
    Conv(X,Y) = E[(X-u)(Y-v)]

    dxi = xi -x_mu
    dyi = yi -y_mu
    Conv(X,Y) = Σ(dxi*dyi)/n
    '''
    mu =  np.mean(data,axis=0)
    mu0 = mu[0]
    mu1 = mu[1]
    return np.sum((data[:,0]-mu0)*(data[:,1]-mu1))/len(data)
def conv2(data):
    '''
    Conv(X,Y)=E[XY]-E[X]E[Y]
    '''
    return np.mean(data[:,0]*data[:,1])-np.mean(data[:,0])*np.mean(data[:,1])

def RouXY(data):
    cv = conv(data)
    v1,v2 = np.std(data[:,0]),np.std(data[:,1])
    return cv/(v1*v2)

data = np.array([[-6,-7],[8,-5],[-4,7],[10,9]])
cv = conv(data)
rouxy = RouXY(data)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data[:,0],data[:,1])
ax.grid()
ax.set_title('Calculate Convariance between Two Variables: X and Y')
ax.set_xlabel('Conv(X,Y) = Σ(dxi*dyi)/n = %d,\ndxi = xi -x_mu,dyi = yi -y_mu'%cv)
#plt.subplots_adjust(bottom=0.1)
plt.tight_layout()

print('Conv result1 = %d'%cv)
print('Conv result2 = %d'%conv2(data))
print('RouXY = %f'%rouxy)
