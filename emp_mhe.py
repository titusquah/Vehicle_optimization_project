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

sp_store=[0]*2
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

gear_gear_eff_est=[0,0]
gear_est=[1,1]
low_vel=[0,0]

gear_gear_eff=[0,0]
gear=[1,1]
br_fudge=[0,0]

tq_est=[0,0]
gop_est=[0,0]
eng_w_est=[0,0]
gb_rat=[0,0]


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
  if eng_w > car.eng_wmax:
      eng_w = car.eng_wmax
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
#mhe_gear_eff=mhe.FV(value=0,lb=0) #gear_efficiency
#mhe_gear_eff.STATUS=0
#mhe_gear_eff.FSTATUS=0
##mhe_gear_eff.DMAX=0.1
#
#
#mhe_gear=mhe.FV(value=0,lb=0.5) #gear
#mhe_gear.STATUS=0 
#mhe_gear.FSTATUS=0
#
#mhe_brake_fudge=mhe.FV(value=1,lb=0,ub=5)#fudge brake
#mhe_brake_fudge.STATUS=0 
#mhe_brake_fudge.FSTATUS=0 #0 if v=0 1 for all other

mhe_eng_fudge=mhe.MV(value=0)#fudge for wheel derivatives
mhe_eng_fudge.STATUS=1
mhe_eng_fudge.FSTATUS=0 

#Manipulated variables
mhe_acc_ped=mhe.MV(0)
mhe_acc_ped.FSTATUS=1
mhe_acc_ped.STATUS=0

mhe_br_ped=mhe.MV(0)
mhe_br_ped.FSTATUS=1
mhe_br_ped.STATUS=0

#Interpolator variable
mhe_eng_wot=mhe.Var(name='eng_wot') #create interpolator variable for eng_wot mhe.Param((car.eng_wmin+car.eng_wmax)/2)
mhe_eng_brk=mhe.Var(name='eng_brk')
mhe_gear_eff=mhe.Var(name='gear_eff')
mhe_gear=mhe.Var(name='gear')
mhe_brake_fudge=mhe.Var(name='br_fudge')
#state variables
mhe_eng_tq=mhe.SV(0)
mhe_gb_opt=mhe.SV(0)
mhe_wh_spt=mhe.SV(0)
mhe_ws=mhe.SV(0)
mhe_gb_op=mhe.SV(0)
mhe_eng_w=mhe.SV(value=car.eng_wmin)
mhe_a=mhe.SV(0)
#sign_chk=mhe.SV(0)#dummy for sign3

#controlled variable
mhe_v=mhe.CV(value=0,lb=0)
#mhe_v.WMODEL=10
mhe_v.FSTATUS=1
mhe_v.STATUS=1
#mhe_v.MEAS_GAP=1

#mhe_eng_w_min_check=mhe.max3(mhe_gb_op*mhe_gear*60/(2*np.pi),car.eng_wmin)
#mhe_eng_w_max_check=mhe.min3(mhe_gb_op*mhe_gear*60/(2*np.pi),car.eng_wmax)
#mhe_eng_w=mhe.min3(mhe_eng_w_min_check,mhe_eng_w_max_check)
mhe_eng_w=mhe.max3(mhe_gb_op*mhe_gear*60/(2*np.pi),car.eng_wmin)
#interpolators
mhe.cspline(mhe_eng_w,mhe_eng_wot,car.eng_spd,car.eng_trq,True)
mhe.cspline(mhe_eng_w,mhe_eng_brk,car.eng_spd,car.eng_brk,True)
mhe.cspline(mhe_v,mhe_gear_eff,car.v_new,car.ge_new,True)
mhe.cspline(mhe_v,mhe_gear,car.v_new,car.gb_new,True)
mhe.cspline(mhe_v,mhe_brake_fudge,car.v_new,car.br_new,True)

#mhe_acc_sign=mhe.sign2(mhe_acc_ped) #1 if acceleration 0 if braking
mhe_acc_sign=mhe.Intermediate(0**(0**mhe_acc_ped)) #1 if pedal 0 if no acc
#mhe_vchck=mhe.sign3(mhe_v-0.1)


#Equations
#mhe.Equation(mhe_acc_ped+mhe_br_ped==sign_chk)
mhe.Equation(mhe_eng_tq==mhe_eng_wot*mhe_acc_ped/100+mhe_eng_brk*(1-mhe_acc_sign))
mhe.Equation(mhe_gb_opt==mhe_eng_tq*mhe_gear_eff*mhe_gear)
mhe.Equation(mhe_wh_spt==mhe_gb_opt*mhe_c7-mhe_eng_fudge)
mhe.Equation(mhe_c2*mhe_ws.dt()==mhe_wh_spt*(mhe_acc_sign)+mhe_c6*mhe_br_ped*mhe_brake_fudge-mhe_c3*mhe_ws**2\
             -mhe_c4*mhe.cos(mhe_grade)*mhe_ws-mhe_c5*mhe.sin(mhe_grade))
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
  time.append(time[-1]+delta_t)
  if i == 5/delta_t:
        sp = 25
  if i == 50/delta_t:
      sp = 0
  if i == 100/delta_t:
      sp = 15
  if i == 150/delta_t:
      sp = 20
  if i == 200/delta_t:
      sp = 10
  if i == 250/delta_t:
      sp = 0
  sp_store.append( sp)
  error = sp - vs[-1]
  es.append(error)
  sum_int = sum_int + error * delta_t
  u = ubias + kc*error + tauI * sum_int


  if u >= 100.0:
      u = 100.0
      sum_int = sum_int - error * delta_t
  if u <= -50:
      u = -50
      sum_int = sum_int - error * delta_t

  if vs[-1] < 0.1 and u < 0:
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
  gb_rat.append(new_pts[1])
  
  gear_gear_eff.append(new_pts[3]*new_pts[2])
  gear.append(new_pts[3])
  if new_pts[10]<=0.1:
    br_fudge.append(0)
  else:
    br_fudge.append(1)
  
#  if i==20:
#    for fixed in (mhe_gear_eff,mhe_gear,mhe_brake_fudge):
#      fixed.DMAX=1
    
  rand_vs.append(float(new_pts[10]))
#  if i%60==0:
#    rand_vs.append(float(new_pts[10])+(np.random.rand()-0.5)*5)
#  else:
#    rand_vs.append(float(new_pts[10])+(np.random.rand()-0.5)*2)
  
  
  mhe_v.MEAS=rand_vs[-1]
#  mhe_gear_eff.MEAS=gear_gear_eff[-1]
#  mhe_gear.MEAS=gear[-1]
#  mhe_brake_fudge.MEAS=br_fudge[-1]
  if u>0:
    mhe_acc_ped.MEAS=float(act_ped[-1])
    mhe_br_ped.MEAS=0.
  elif u<0:
    
    mhe_acc_ped.MEAS=0.
    mhe_br_ped.MEAS=float(act_ped[-1])
    
  else:
    mhe_acc_ped.MEAS=0.
    mhe_br_ped.MEAS=0.
#  if i==11:
#    mhe_eng_fudge.STATUS=1
#    for fixed in (mhe_eng_fudge,mhe_gear_eff,mhe_gear,mhe_brake_fudge):
#      fixed.STATUS=1

  try:
    mhe.solve()
    est_v.append(mhe_v.MODEL)
    
#    gear_gear_eff_est.append(mhe_gear_eff.NEWVAL)
#    gear_est.append(mhe_gear.NEWVAL)
#    low_vel.append(mhe_brake_fudge.NEWVAL)
    
    gear_gear_eff_est.append(mhe_gear_eff[-1])
    gear_est.append(mhe_gear[-1])
    low_vel.append(mhe_brake_fudge[-1])
    
    
    tq_est.append(mhe_eng_tq.MODEL)
    gop_est.append(mhe_gb_op.MODEL)
    eng_w_est.append(mhe_eng_w[-1])
  except Exception as e:
    est_v.append(est_v[-1])
    
    gear_gear_eff_est.append(gear_gear_eff_est[-1])
    gear_est.append(gear_est[-1])
    low_vel.append(low_vel[-1])
    
    tq_est.append(tq_est[-1])
    gop_est.append(gop_est[-1])
    eng_w_est.append(eng_w_est[-1])
    print("Failed: {0}".format(e))
#    break
  if i%50==0:
    print(i)
    plt.clf() 
    plt.subplot(311)
    plt.plot(time,est_v,label='Model')
    plt.plot(time,rand_vs,'x',label='Measured')
    plt.ylabel('Velocity (m/s)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc=2,fontsize=16)
    plt.subplot(312)
    plt.plot(time,gear_gear_eff_est,label=r'Gear$\times$gear efficiency')
    plt.plot(time,gear_est,label='Gear')
    plt.plot(time,low_vel,label='V<0.1 (brake fudge)')
    plt.legend(loc=2,fontsize=16)
    plt.ylabel('Guessed Value',fontsize=16)
    plt.xlabel('Time (s)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplot(313)
    plt.plot(time,np.array(act_ped)/100,label=r'Brake/Acceleration')
    plt.plot(time,gb_rat,label='Gear')
    plt.plot(time,np.array(tq_est)/100,label='Torque/100')
    plt.plot(time,np.array(gop_est)/100,label='Gear box opt/100')
    plt.plot(time,np.array(eng_w_est)/1000,label='Engine speed /1000')
    plt.legend(loc=2,fontsize=16)
    plt.ylabel('Estimated Value',fontsize=16)
    plt.xlabel('Time (s)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.draw()
    plt.pause(0.05) 
  uold=u
  
  


#plt.figure()
##plt.plot(time,est_v,label='Model')
#plt.plot(time,sp_store,label='SP')
#plt.plot(time,vs,'x',label='Actual')
#plt.legend(loc='best')
#plt.show()
    
    
        
    
    

    






