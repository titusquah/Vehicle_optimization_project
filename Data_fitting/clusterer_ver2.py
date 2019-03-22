import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


#%% Conversions
m_per_mi=1609.34 #meters/mile
s_per_hr=3600 #seconds/hour
#%%Retrieve data

fnums=['56{0}'.format(i) for i in [39,42,43,44,45,48]]
acc_time=[]
acc_mps=[]
acc_gph=[]
acc_mps2=[]

filtered_mps=[]
filtered_time=[]
filtered_gph=[]

for fnum in fnums:
  df=pd.read_excel(r"C:\Users\tq220\Documents\Tits things\2018-2019\Dynamic Optimization\Final Project\IMG_{0}_data.xlsx".format(fnum))
  
  time=df['t(s)'].values
  mph=df['V (mph)'].values
  mpg=df['Consumption (mpg)'].values
  gph=mph/mpg #gal/hr
  mps=mph*m_per_mi/s_per_hr #mi/hr * m/mi *hr/sec=m/s
  
  for i in range(len(time)-1):
    if (mps[i+1]-mps[i])/(time[i+1]-time[i])>0:
      acc_mps2+=[(mps[i+1]-mps[i])/(time[i+1]-time[i])]
      acc_time+=[np.mean([time[i+1],time[i]])]
    
      acc_mps+=[np.mean([mps[i+1],mps[i]])]
      acc_gph+=[np.mean([gph[i+1],gph[i]])]
      
      filtered_mps+=[mps[i+1],mps[i]]
      filtered_time+=[time[i+1],time[i]]
      filtered_gph+=[gph[i+1],gph[i]]
    



#%% Clustering
X=np.array([acc_mps,acc_gph]).transpose()
agg_cluster_model = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=3)
db_model = DBSCAN(eps=1.1, min_samples=3)
    

ypred=KMeans(n_clusters=3).fit_predict(X)

#
#X=np.array([acc_mps2,acc_gph]).transpose()
#agg_cluster_model = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=3)
#db_model = DBSCAN(eps=1.1, min_samples=3)

#ypred=KMeans(n_clusters=3).fit_predict(X)
#ypred=agg_cluster_model.fit_predict(X)
#ypred=db_model.fit_predict(X)
#%%plot
plt.close('all')
#3d plot of data
#fig,ax=plt.subplots()
#plot1=ax.scatter(acc_mps,acc_gph,c=acc_mps2,cmap='viridis')
#ax.set_xlabel('Velocity (m/s)')
#ax.set_ylabel('Fuel consumption (gal/hr)')
#ax.grid(True)
#cbar = plt.colorbar(plot1)
#cbar.set_label(r'Acceleration (m/s$^2$)', rotation=270,labelpad=15)
#plt.show()  

#3d cluster plotting
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(acc_mps, acc_gph, acc_mps2)
#ax.set_xlabel('Velocity (m/s)')
#ax.set_ylabel('Fuel consumption (gal/hr)')
#ax.set_zlabel(r'Acceleration (m/s$^2$)')
#plt.show()

#2d cluster plotting
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.scatter(acc_mps, acc_gph,c=ypred,cmap='brg')
ax.set_xlabel('Velocity (m/s)')
#ax.set_xlabel(r'Acceleration (m/s$^2$)')
ax.set_ylabel('Fuel consumption (gal/hr)')
plt.show()
#%%Save to csv.
df_send=pd.DataFrame({'time':filtered_time,'mps':filtered_mps,'gph':filtered_gph})
df_send.to_csv('orig_vals_for_cluster_data.csv',index=False)

