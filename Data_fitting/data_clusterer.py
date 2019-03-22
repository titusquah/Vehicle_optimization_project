# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:53:29 2019

@author: Titus
"""

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
plt.style.use('ggplot')
cmap = ListedColormap(["#e41a1c","#984ea3","#a65628","#377eb8","#ffff33","#4daf4a","#ff7f00"])

df=pd.read_excel(r"C:\Users\tq220\Documents\Tits things\2018-2019\Dynamic Optimization\Final Project\Fit_data.xlsx")

X=np.array([df.k1,df.k2,df.k3,df['Start V (m/s)']]).transpose()
ypred=KMeans(n_clusters=3).fit_predict(X)

plt.close('all')
fig,ax=plt.subplots(3,sharex=True)
ax[0].scatter(X[:,3],X[:,0],c=ypred,marker='o',cmap=cmap)
ax[0].grid(True)
ax[1].scatter(X[:,3],X[:,1],c=ypred,marker='o',cmap=cmap)
ax[1].grid(True)
ax[2].scatter(X[:,3],X[:,2],c=ypred,marker='o',cmap=cmap)
ax[2].grid(True)
plt.show()