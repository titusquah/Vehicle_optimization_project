import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from gekko import GEKKO
import car_v04 as car

#%%Parameters
make_mp4 = False

#Simulation time step definition
tf        = 300                 #final time for simulation
nsteps    = 3001                 #number of time steps
delta_t   = tf / (nsteps - 1)   #length of each time step
ts        = np.linspace(0,tf,nsteps)
#Advanced cyber driver
step      = [0]   #assigning array for pedal position
#step[11:] = 75.0               #75% @ timestep 11
#step[40:] = -50                #-50% @ timestep 40 to simulate braking
ubias     = 0.0
kc        = 15.0
tauI      = 09.0
sum_int   = 0
sp        = 0
gear      = 1


grades=np.zeros(nsteps)

sp_store=[25]*2
vs=[0]*2
rand_vs=[0]*2
es=[0]*2
ies=[0]*2
act_ped=[0]*2
eng_w = [car.eng_wmin]*2
ws=[0]*2
gb_op=[0]*2
grade=[0]*2
est_v=[0,0]
time=[0,0]
fails=0

cd=[0.003]*2
rr1=[0.001]*2
rr2=[0.3]*2
#mhe_tau=mhe.FV(value=0.001)
k1=[10]*2
k2=[10]*2
tau=[0.001]*2

#%%Vehicle simulator
def vehicle(u,eng_w,v0,ws,delta_t,gb_op,grade): #ws  and gb_op are lists of len 2
  if u > 0:
    eng_tq= car.eng_wot(eng_w,u) * u/100
  else:   
    eng_tq= car.eng_wot(eng_w,u)
  gb_rat, gb_eff = car.g_box(v0,u)
  gear= car.gb[int(gb_rat)]
  gb_opt=eng_tq * gear * gb_eff- (car.Igb_o + car.Igb_i) * (gb_op[1] - gb_op[0])/delta_t
  wh_spt= gb_opt * car.Fdr * car.Fef- (car.wh_inf + car.wh_inr) * (ws[1] - ws[0])/delta_t
  wh_spd= odeint(car.roadload, ws[1], [0,delta_t], args=(wh_spt,u,grade,v0))
  ws=wh_spd[-1]
  gb_op=ws*car.Fdr
  gb_ip=gb_op* gear
  eng_w=gb_ip* 60 / (2 * np.pi) #eng_sp
  if eng_w < car.eng_wmin:
    eng_w = car.eng_wmin
  vs= ws * car.wh_rd
  acc = (vs - v0) / (delta_t * 9.81)
        
  return [eng_tq, #0
          gb_rat, #1
          gb_eff, #2
          gear,   #3
          gb_opt, #4
          wh_spt, #5
          ws,     #6
          gb_op,  #7
          gb_ip,  #8
          eng_w,  #9
          vs,     #10
          acc]    #11

#%%Gekko mhe
mhe=GEKKO()
mhe.time=np.linspace(0,2,21) 
mhe_ac=mhe.MV(value=0,lb=0,name='ac')
mhe_ac.STATUS=0
mhe_ac.FSTATUS=1

mhe_br=mhe.MV(value=0,ub=0,name='br')
mhe_ac.STATUS=0
mhe_ac.FSTATUS=1


mhe_v = mhe.CV(value=0, name='v',lb=0) #vs
mhe_v.FSTATUS=1
mhe_v.STATUS=1
mhe_v.MEAS_GAP=1

mhe_a = mhe.SV(value=0,name='a')          #acc
mhe_f=mhe.SV(value=0,name='f')

mhe_grade=mhe.MV(0)
mhe_grade.STATUS=0
mhe_grade.FSTATUS=1

mhe_cd=mhe.FV(value=0.003875,lb=0,ub=50)
mhe_rr1=mhe.FV(value=0.001325,lb=0,ub=50)
mhe_rr2=mhe.FV(value=0.265,lb=0,ub=50)
mhe_tau=mhe.FV(value=0.01,lb=0)
mhe_k1=mhe.FV(value=10)
mhe_k2=mhe.FV(value=0.2)
mhe_tm=mhe.Param(value=360) # total mass
mhe_iw=mhe.Param(value=26) # total mass

#dmaxs=[10,10,10,]
for ind,fixed in enumerate([mhe_cd,mhe_rr1,mhe_rr2,mhe_k1,mhe_k2,mhe_tau]):
  fixed.STATUS=0
  fixed.FSTATUS=0
  fixed.DMAX=10
#  fixed.LOWER=0
mhe_k1.DMAX=100
mhe_k2.DMAX=200

#mhe.Equation(mhe_tau*mhe_f.dt()+mhe_f == mhe_k1*mhe_u)
mhe.Equation(mhe_tau*mhe_f.dt()+mhe_f == mhe_k1*mhe_ac+mhe_k2*mhe_br)
mhe.Equation(mhe_tm*mhe_a==mhe_f-mhe_cd*mhe_v**2- mhe_rr1*mhe.cos(mhe_grade)*mhe_v- mhe_rr2*mhe.sin(mhe_grade))
mhe.Equation(mhe_v.dt()==mhe_a)

# set up the solver
mhe.options.IMODE = 5
mhe.options.SOLVER = 1
mhe.options.EV_TYPE=1
mhe.options.COLDSTART=1
mhe.options.AUTO_COLD=2


# set up the objective

#%% Simulate


plt.close('all')
plt.figure(figsize=(10,7))
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.ion()
plt.show()

#Simulation 
for i in range(nsteps - 1):
  time.append(time[-1]+delta_t)
  if i == 5/car.delta_t:
      sp = 25
  if i == 50/car.delta_t:
      sp = 0
  if i == 100/car.delta_t:
      sp = 15
  if i == 150/car.delta_t:
      sp = 20
  if i == 200/car.delta_t:
      sp = 10
  if i == 250/car.delta_t:
      sp = 0
  sp_store.append(sp)
  error = sp - vs[-1]
  es.append( error)
  sum_int = sum_int + error * car.delta_t
  u = ubias + kc*error + tauI * sum_int
  ies.append(sum_int)
  step.append(u)
  if u >= 100.0:
      u = 100.0
      sum_int = sum_int - error * car.delta_t
  if u <= -50:
      u = -50
      sum_int = sum_int - error * car.delta_t

  if vs[i] < 0.1 and u < 0:
      act_ped.append(-50)
  else:
      act_ped.append(u)
  grade.append(grades[i])
  new_pts=vehicle(u,eng_w[-1],vs[-1],ws[-2:],delta_t,gb_op[-2:],grade[-1]) #[eng_tq, #0
#                                                                      gb_rat, #1
#                                                                      gb_eff, #2
#                                                                      gear,   #3
#                                                                      gb_opt, #4
#                                                                      wh_spt, #5
#                                                                      ws,     #6
#                                                                      gb_op,  #7
#                                                                      gb_ip,  #8
#                                                                      eng_w,  #9
#                                                                      vs,     #10
#                                                                      acc]    #11
  eng_w.append(new_pts[9])
  ws.append(new_pts[6])
  gb_op.append(new_pts[7])
  vs.append(new_pts[10])
  if i%60==0:
    rand_vs.append(float(new_pts[10])+(np.random.rand()-0.5)*5)
  else:
    rand_vs.append(float(new_pts[10])+(np.random.rand()-0.5)*1.2)
  
  for ind,fixed in enumerate([mhe_cd,mhe_rr1,mhe_rr2,mhe_k1,mhe_k2,mhe_tau]):
    
    if i>21 and i%6==ind and vs[-1]>0.1  :
      fixed.STATUS=1
    else:
      fixed.STATUS=0
  
  mhe_v.MEAS=float(rand_vs[-1])
  mhe_grade.MEAS=grade[-1]
  if u<0:
    mhe_ac.MEAS=0
    mhe_br.MEAS=u
#    mhe_k1.STATUS=0
#    mhe_k2.STATUS=1
  else:
    mhe_ac.MEAS=u
    mhe_br.MEAS=0
#    mhe_k1.STATUS=1
#    mhe_k2.STATUS=0
#  if i>21 :
#    for fixed in (mhe_cd,mhe_rr1,mhe_rr2,mhe_k1):
#      fixed.STATUS=1
    
  
    
  if i==101:
    for fixed in (mhe_cd,mhe_rr1,mhe_rr2):
      fixed.DMAX=10

  try:
    
    mhe.solve(disp=False)
    est_v.append(mhe_v.MODEL)
    cd.append(mhe_cd.NEWVAL)
    rr1.append(mhe_rr1.NEWVAL)
    rr2.append(mhe_rr2.NEWVAL)
    k1.append(mhe_k1.NEWVAL)
    k2.append(mhe_k2.NEWVAL)
    tau.append(mhe_tau.NEWVAL)
    
  except KeyboardInterrupt:
    print("Stopping")
    break
  except Exception as e:
    fails+=1
    est_v.append(est_v[-1])
    
    cd.append(cd[-1])
    rr1.append(rr1[-1])
    rr2.append(rr2[-1])
    k1.append(k1[-1])
    k2.append(k2[-1])
    tau.append(tau[-1])
    print('Failed: {0}'.format(e))
  
#  if i%50==0:
#    print(i)
#    print('# of fails: {0}'.format(fails))
#    fails=0
  plt.clf() 
  plt.subplot(311)
  
  plt.plot(time,rand_vs,'x',label='Actual',alpha=0.5)
  plt.plot(time,est_v,label='Model')
  plt.legend(loc=2)
  plt.subplot(312)
  plt.plot(time,cd,label="Drag coeff")
  plt.plot(time,rr1,label="Rolling resistance 1")
  plt.plot(time,rr2,label="Rolling resistance 2")
  plt.plot(time,k1,label="Acc Gain")
  plt.plot(time,k2,label="Br Gain")
  plt.plot(time,tau,label="Time constant")
  plt.legend(loc=2)
  plt.subplot(313)
  plt.plot(time,act_ped,label='Pedal')
  plt.legend(loc=2)
  
  plt.draw()
  plt.pause(0.05) 
#plt.figure()
#plt.plot(time,est_v,label='Model')
#plt.plot(time,vs,'x',label='Actual')
#plt.legend(loc='best')
#plt.show()
    
    
        
    
    

    






