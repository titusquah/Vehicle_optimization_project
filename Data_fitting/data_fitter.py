# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:01:36 2019

@author: Titus
"""

#import statsmodels.formula.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
plt.style.use('ggplot')
plt.close('all')
start=0

img=5648
data_1=pd.read_excel(r"C:\Users\tq220\Documents\Tits things\2018-2019\Dynamic Optimization\Final Project\IMG_{}_data.xlsx".format(img),header=0)

#%% load data in
#data_1.plot(x='t(s)')
#plt.grid(True)
#plt.show()
tm=np.array(data_1['t(s)'])
xpos=np.array(data_1['Position (mi)'])
mph=np.array(data_1['V (mph)'])
mpg=np.array(data_1['Consumption (mpg)'])
gph=mph/mpg #gal/hr
atm=np.array([(tm[i+1]+tm[i])/2 for i in range(len(tm)-1)])
mphps=np.array([(mph[i+1]-mph[i])/(tm[i+1]-tm[i]) for i in range(len(tm)-1)]) #mi/hr/s

#%% Conversions

m_per_mi=1609.34 #meters/mile
s_per_hr=3600 #seconds/hour

xpos=xpos*m_per_mi #mi * m/mi
mps=mph*m_per_mi/s_per_hr #mi/hr * m/mi *hr/sec=m/s
mps_2=np.array([(mps[i+1]-mps[i])/(tm[i+1]-tm[i]) for i in range(len(tm)-1)]) #m/s^2

rar=mps_2>=0

t=[]
x=[]
v=[]
a=[]
at=[]
cons=[]
i=start
try:
  while mps_2[i]>=0:
    t.append(tm[i])
    x.append(xpos[i])
    v.append(mps[i])
    a.append(mps_2[i])
    at.append(atm[i])
    cons.append(gph[i])
    i+=1
except:
  t=tm[start:]
  x=xpos[start:]
  v=mps[start:]
  a=mps_2[start:]
  at=atm[start:]
  cons=gph[start:]
  
  
t=np.array(t)
x=np.array(x)
v=np.array(v)
a=np.array(a)
at=np.array(at)
cons=np.array(cons)

#%%
init_guess=np.array([1400,4,1,1,1])

param_bounds=([1450,0.36,-np.inf,-np.inf,-np.inf],[1630,100,np.inf,np.inf,np.inf])

def car_velocity(X,m,C,k1,k2,k3):
  gph,mps=X
  fe=k1*gph+k2/(gph+1)+k3*np.sqrt(gph)
  fd=C*mps**2
  fn=fe-fd
  acc=fn/m
  return mps+acc
def r_2(actual_y,predicted_y):
  actual_y=np.array(actual_y)
  predicted_y=np.array(predicted_y)
  num=sum((actual_y-predicted_y)**2)
  den=sum((actual_y-np.mean(actual_y))**2)
  r_2=(1-num/den)
  return r_2

p_opt,p_cov=curve_fit(car_velocity,(cons,v),v,bounds=param_bounds)
#def velocity
#def fit_function(cons,m,C,k1,k2,k3):
#  return quad(car_forces,0,max(t))




fig,ax=plt.subplots(nrows=4,sharex=True)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

pred_v=car_velocity((np.array(cons),np.array(v)),p_opt[0],p_opt[1],p_opt[2],p_opt[3],p_opt[4])
pred_x=[trapz(pred_v[0:i])+x[0] for i in range(len(t))]
pred_a=np.array([(pred_v[i+1]-pred_v[i])/(t[i+1]-t[i]) for i in range(len(t)-1)]) #m/s^2

ax[0].plot(t,x,'C0o',label='Measured')
ax[0].plot(t,pred_x,'C1-',label='Predicted')
ax[0].set_ylabel('Position (m)')


ax[1].plot(t,v,'C0o',label='Measured')
ax[1].plot(t,pred_v,'C1-',label='Predicted')
ax[1].set_ylabel(r'Velocity ($\frac{m}{s}$)')

ax[2].plot(at[:-1],a[:-1],'C0o',label='Measured')
ax[2].plot(at[:-1],pred_a,'C1-',label='Predicted')
ax[2].set_ylabel(r'Acceleration ($\frac{m}{s^2}$)')

ax[3].plot(t,cons,'C0o',label='Measured')

ax[3].set_ylabel(r'Fuel Consumption ($\frac{gal}{hr}$)')

ax[3].set_xlabel("Time (sec)")
plt.pause(3)
plt.savefig('Results_IMG{0}_{1}_start.png'.format(img,start))

plt.show()

param_names=['m','C','k1','k2','k3']

print(r_2(v,pred_v),v[0])
for i,val in enumerate(p_opt):
  print("{0}={1:.2f}".format(param_names[i],val))
