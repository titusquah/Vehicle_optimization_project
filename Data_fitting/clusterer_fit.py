# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:24:17 2019

@author: Titus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit

acc_df=pd.read_csv(r"C:\Users\tq220\Documents\Tits things\2018-2019\Dynamic Optimization\Final Project\cluster_data.csv")
v_df=pd.read_csv(r"C:\Users\tq220\Documents\Tits things\2018-2019\Dynamic Optimization\Final Project\cluster_data_orig_vals.csv")
v_df['class']=[acc_df['class'][int(i/2)]for i in range(len(v_df))]

param_bounds=([1450,0.1,-np.inf,-np.inf,-np.inf],[1630,100,np.inf,np.inf,np.inf])

def car_velocity(X,m,C,k1,k2,k3):
  gph,mps,ti,tf=X
  fe=k1*gph+k2/(gph+1)+k3*np.sqrt(gph)
  fd=C*mps**2
  fn=fe-fd
  acc=fn/m
  return mps+acc*(tf-ti)
def r_2(actual_y,predicted_y):
  actual_y=np.array(actual_y)
  predicted_y=np.array(predicted_y)
  num=sum((actual_y-predicted_y)**2)
  den=sum((actual_y-np.mean(actual_y))**2)
  r_2=(1-num/den)
  return r_2

m=[]
C=[]
k1=[]
k2=[]
k3=[]


for i in range(v_df['class'].nunique()):
  df=v_df[v_df['class']==i].reset_index(drop=True)
  print(min(df['mps']),max(df['mps']))
  df_i=df.iloc[::2]
  df_f=df.iloc[1::2]
  cons=df_f['gph'].values
  vi=df_i['mps'].values
  vf=df_f['mps'].values
  ti=df_i['time'].values
  tf=df_f['time'].values
  p_opt,p_cov=curve_fit(car_velocity,(cons,vi,ti,tf),vf,bounds=param_bounds)
  
  m.append(p_opt[0])
  C.append(p_opt[1])
  k1.append(p_opt[2])
  k2.append(p_opt[3])
  k3.append(p_opt[4])

def car_acc(mps,gph):
  if mps>=0 and mps<11.064:
    cl=1
  elif mps>=11.064 and mps<22.3519:
    cl=0
  else:
    cl=2
  fe=k1[cl]*gph+k2[cl]/(gph+1)+k3[cl]*np.sqrt(gph)
  fd=C[cl]*mps**2
  fn=fe-fd
  acc=fn/m[cl]
  return acc

pred_acc=[car_acc(v_df['mps'][2*i],v_df['gph'][2*i+1]) for i in range(len(acc_df))]
acc_df['pred_mps2']=pred_acc
print(r_2(acc_df['mps2'].values,pred_acc))