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
mhe.time=np.linspace(0,1,11) 

#parameters
mhe_grade=mhe.Param(0)
mhe_c1=mhe.Param(car.wh_rd) #wheel radius
mhe_c2=mhe.Param(car.Iw) #inertia
mhe_c3=mhe.Param(0.5*car.rho*car.Cd*car.A*car.wh_rd**3) #drag coeff
mhe_c4=mhe.Param(car.wh_rd*car.Crr*(car.m+car.load)) #rolling resist
mhe_c5=mhe.Param(car.wh_rd*(car.m+car.load)*9.81) #gravity
mhe_c6=mhe.Param(car.Fb*car.wh_rd)
mhe_c7=mhe.Param(car.Fdr * car.Fef)
mhe_c8=mhe.Param(car.wh_inf + car.wh_inr)
mhe_c9=mhe.Param(car.Igb_o + car.Igb_i)
mhe_c10=mhe.Param(car.Fdr)

#Shifting Parameters
mhe_p1=mhe.FV(value=0)
mhe_p1.STATUS=0
mhe_p1.FSTATUS=0

mhe_p2=mhe.FV(value=0)
mhe_p2.STATUS=0
mhe_p2.FSTATUS=0

#Manipulated variables
mhe_acc_ped=mhe.MV(0)
mhe_acc_ped.FSTATUS=1
mhe_acc_ped.STATUS=0

mhe_br_ped=mhe.MV(0)
mhe_br_ped.FSTATUS=1
mhe_br_ped.STATUS=0

#Interpolator variable
mhe_eng_wot=mhe.Var(name='eng_wot') #create interpolator variable for eng_wot mhe.Param((car.eng_wmin+car.eng_wmax)/2)

#state variables
mhe_eng_tq=mhe.SV(0)
mhe_gb_opt=mhe.SV(0)
mhe_wh_spt=mhe.SV(0)
mhe_ws=mhe.SV(0)
mhe_gb_op=mhe.SV(0)
mhe_eng_w=mhe.SV(value=car.eng_wmin,lb=car.eng_wmin)
mhe_a=mhe.SV(0)

#controlled variable
mhe_v=mhe.CV(value=0)
mhe_v.FSTATUS=1
mhe_v.STATUS=1
#mhe_v.MEAS_GAP=1

mhe_eng_w=mhe.max3(mhe_gb_op*mhe_p2*60/(2*np.pi),car.eng_wmin)
#mhe_eng_w_max_check=mhe.min3(mhe_gb_op*mhe_p2*60/(2*np.pi),car.eng_wmax)
#mhe_eng_w=mhe.min3(mhe_eng_w_min_check,mhe_eng_w_max_check)
#interpolators
mhe.cspline(mhe_eng_w,mhe_eng_wot,car.eng_spd,car.eng_trq,True)
mhe_acc_sign=mhe.Intermediate(0**(0**mhe_acc_ped)) #1 if pedal is on 0 if pedal is off
mhe_br_sign=mhe.Intermediate(0**(0**-mhe_br_ped))

#Equations
mhe.Equation(mhe_eng_tq==mhe_eng_wot*mhe_acc_ped/100+mhe_eng_wot*mhe_br_sign)
mhe.Equation(mhe_gb_opt==mhe_eng_tq*mhe_p1)
mhe.Equation(mhe_wh_spt==mhe_gb_opt*mhe_c7)
mhe.Equation(mhe_ws==mhe_wh_spt*mhe_acc_sign+mhe_c6*mhe_br_ped-mhe_c3*mhe_ws**2-mhe_c4*mhe.cos(mhe_grade)-mhe_c5*mhe.sin(mhe_grade))
mhe.Equation(mhe_gb_op==mhe_ws*mhe_c10)

mhe.Equation(mhe_v==mhe_ws*mhe_c1)
mhe.Equation(mhe_v.dt()==mhe_a)
#mhe.Equation(mhe_acc_ped*mhe_br_ped==0)

# set up the solver
mhe.options.IMODE = 5
mhe.options.SOLVER = 1
mhe.options.EV_TYPE=2
mhe.options.COLDSTART=1
mhe.options.AUTO_COLD=1



#%% Simulate


plt.close('all')
plt.figure(figsize=(10,7))
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.ion()
plt.show()

#Simulation 
for i in range(nsteps - 1):
  try:
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
  #  grade.append(grades[i])
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
    
    mhe_v.MEAS=float(new_pts[10])
    if u>0:
      mhe_acc_ped.MEAS=u
      mhe_br_ped.MEAS=0
    elif u<0:
      mhe_acc_ped.MEAS=0
      mhe_br_ped.MEAS=u
    else:
      mhe_acc_ped.MEAS=0
      mhe_br_ped.MEAS=0
    if i==11:
      for fixed in (mhe_p1,mhe_p2):
        fixed.STATUS=1
  
    try:
      mhe.solve()
      est_v.append(mhe_v.MODEL)
    except:
      est_v.append(est_v[-1])
      break
    if i%50==0:
      plt.clf() 
      plt.plot(time,est_v,label='Model')
      plt.plot(time,vs,'x',label='Actual')
      plt.draw()
      plt.pause(0.05) 
  except KeyboardInterrupt:
    print("exiting")
    break
#plt.figure()
#plt.plot(time,est_v,label='Model')
#plt.plot(time,vs,'x',label='Actual')
#plt.legend(loc='best')
#plt.show()
    
    
        
    
    

    






