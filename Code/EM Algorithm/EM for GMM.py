# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
import pandas as pd
#mpl.rcParams['font.sans-serif'] = [u'SimHei']
#mpl.rcParams['axes.unicode_minus'] = False

#%%
def visdata(g,b,h):
    global gga,bga,g1,g2,u1,u2,sigma1,sigma2
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(121)
    # plot the ground true data distributions
    ax.hist(h,bins=20,alpha=0.5,label='total')
    ax.hist(g,bins=20,alpha=0.5,label='girls')
    ax.hist(b,bins=20,alpha=0.5,label='boys')
    ax.set_xlabel("height")
    ax.set_title('Histogram for Heights')
    ax.legend(loc='lower right',fontsize=7)
    ax.grid()
    
    ax = fig.add_subplot(122)
    # plot two gauss distributions
    ax.plot(np.sort(g),norm.pdf(np.sort(g),u1,sigma1),label='Gauss-1')
    ax.plot(np.sort(b),norm.pdf(np.sort(b),u2,sigma2),label='Gauss-2')
    # plot the mixture gaussian distributions
    ax.scatter(np.sort(g),g1*norm.pdf(np.sort(g),u1,sigma1)+g2*norm.pdf(np.sort(g),u2,sigma2),c='r',s=1)
    ax.scatter(np.sort(b),g1*norm.pdf(np.sort(b),u1,sigma1)+g2*norm.pdf(np.sort(b),u2,sigma2),c='r',s=1)
    # plot sample weight for Gauss 1 and 2
    ax.scatter(h,gga/10,s=1,label='responsibility1/10')
    ax.scatter(h,bga/10,s=1,label='responsibility2/10')
    ax.set_xlabel("height")
    ax.set_title('Parameters')
    ax.legend(loc='lower right',fontsize=7)
    ax.grid()

    
def averageWeight(height_,ggamma_,gn_):
    return sum(height_*ggamma_)/gn_

def varianceWeight(height_,ggamma_,gmu_,gn_):
    return np.sqrt(sum(ggamma_*(height_-gmu_)**2)/gn_)

def gauss(x_,gmu_,gsigma_):
    return 1.0/(np.sqrt(2*np.pi)*gsigma_)*np.exp(-(x_-gmu_)**2/(2*(gsigma_)**2))
    
#%%
girl_num = 500
boy_num = 400
girl_mean = 160
boy_mean = 170
girl_std = 4
boy_std = 5

np.random.seed(1)
boy_hight = np.random.normal(size = boy_num) * boy_std + boy_mean
girl_hight = np.random.normal(size = girl_num) * girl_std + girl_mean
height = np.concatenate((boy_hight,girl_hight)).reshape(-1,1)
  
 
#%%
N = len(height)
gp = 0.5 # girl probability
bp = 0.5 # boy probability
gmu, gsigma = min(height),4
bmu, bsigma = max(height),5
#    ggamma = range(N)
#    bgamma = range(N)
ggamma = np.ones((N,1))
bgamma = np.ones((N,1))
cur = [gp,bp,gmu,gsigma,bmu,bsigma]
now = []

#%%
times = 0

while times<100:
    i = 0
    for x in height:
        ggamma[i] = gp*gauss(x,gmu,gsigma) #probability-sample i belongs to girl
        bgamma[i] = bp*gauss(x,bmu,bsigma) #probability-sample i belongs to boy
        s = ggamma[i]+bgamma[i]
        ggamma[i]/=s
        bgamma[i]/=s
        i = i+1
        
    gn = sum(ggamma)
    gp = float(gn)/float(N)
        
    bn = sum(bgamma)
    bp = float(bn)/float(N)
    
    gmu = averageWeight(height,ggamma,gn)
    gsigma = varianceWeight(height,ggamma,gmu,gn)
    
    bmu = averageWeight(height,bgamma,bn)
    bsigma = varianceWeight(height,bgamma,bmu,bn)
    
    now = [gp,bp,gmu,gsigma,bmu,bsigma]
    
#        if isSame(cur,now):
#            break
    cur = now
    times +=1
print('Girl average Height:%f'%gmu)
print('Boy  average Height:%f'%bmu)
print('Girl Height Var:%f'%gsigma)
print('Boy  Height Var:%f'%bsigma)
#%%

global gga,bga,g1,g2,u1,u2,sigma1,sigma2
gga = ggamma
bga = bgamma
g1 = gp
g2 = bp
u1 = gmu
u2 = bmu
sigma1 = gsigma
sigma2 = bsigma
#%%
visdata(girl_hight,boy_hight,height)
#%%
#g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-9, max_iter=500)
#g.fit(height)
#print ('类别概率:\t', g.weights_[0])
#print ('均值:\n', g.means_, '\n')
#print ('方差:\n', g.covariances_, '\n')

