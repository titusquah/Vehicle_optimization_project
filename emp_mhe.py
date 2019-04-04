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
es=[0]*2
ies=[0]*2
act_ped=[0]*2
eng_w = [car.eng_wmin]*2
ws=[0]*2
gb_op=[0]*2
grade=[0]*2
est_v=[0,0]
time=[0,0]

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
  wh_spd= odeint(car.roadload, ws[1], [0,delta_t], args=(wh_spt,u,grade))
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
mhe=GEKKO(remote=False,server='http://127.0.0.1')
mhe.time=np.linspace(0,60,61) 
mhe_u=mhe.MV(value=0,name='u')
mhe_u.STATUS=0
mhe_u.FSTATUS=1

mhe_v = mhe.CV(value=0, name='v') #vs
mhe_v.FSTATUS=1
mhe_v.STATUS=1
#mhe_v.MEAS_GAP=1

mhe_a = mhe.SV(value=0,name='a')          #acc
mhe_f=mhe.SV(value=0,name='f')

mhe_grade=mhe.MV(0)
mhe_grade.STATUS=0
mhe_grade.FSTATUS=1

mhe_cd=mhe.FV(value=5)
mhe_rr1=mhe.FV(value=5)
mhe_rr2=mhe.FV(value=5)
mhe_tau=mhe.FV(value=1)
mhe_k1=mhe.FV(value=10)
mhe_tm=mhe.FV(value=200) # total mass


for fixed in (mhe_cd,mhe_rr1,mhe_rr2,mhe_tau,mhe_k1):
  fixed.STATUS=1
  fixed.FSTATUS=0
  fixed.LOWER=0
  
mhe.Equation(mhe_tau * mhe_f.dt() + mhe_f == mhe_k1*mhe_u)
mhe.Equation(mhe_tm*mhe_a==mhe_f-mhe_cd*mhe_v**2- mhe_tm*mhe_rr1*mhe.cos(mhe_grade)*mhe_v- mhe_tm*mhe_rr2*9.81*mhe.sin(mhe_grade))
mhe.Equation(mhe_v.dt()==mhe_a)

# set up the solver
mhe.options.IMODE = 5
mhe.options.SOLVER = 3
mhe.options.EV_TYPE=2
mhe.options.COLDSTART=1


# set up the objective

#%% Simulate


plt.close('all')
#plt.figure(figsize=(10,7))
#mng = plt.get_current_fig_manager()
#mng.window.showMaximized()
#plt.ion()
#plt.show()

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
  
  mhe_v.MEAS=new_pts[10]
  mhe_grade.MEAS=grade[-1]
  mhe_u.MEAS=u
  mhe.solve(disp=False)
  
  est_v.append(mhe_v.MODEL)
  
#  plt.clf()
#  plt.plot(time,est_v,label='Model')
#  plt.plot(time,vs,'x',label='Actual')
#  plt.draw()
#  plt.pause(0.05) 
plt.figure()
plt.plot(time,est_v,label='Model')
plt.plot(time,vs,'x',label='Actual')
plt.legend(loc='best')
plt.show()
    
    
        
    
    

    






