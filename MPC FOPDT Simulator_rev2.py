import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import car_v04 as car # this is vehicle_model_v04.py in git
from scipy.integrate import odeint
from datetime import datetime
#%%
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
  ws=float(wh_spd[-1])
  ws=max([0,ws])
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
#%%
xgoals=[10.,50.,100.]
xldb=1.
xhdb=0.5
speed_limit=30.
final=np.zeros(31)
final[-1]=1
# set up the gekko models
mhe = GEKKO(name='mhe')
mpc = GEKKO(name='mpc')

# set up the empirical equations relating pedal% to velocity
for m in (mhe, mpc):
  
  # time
  m.t = m.Var()
  
  # variables to keep track of
  m.a = m.SV(value=0)
  m.v = m.CV(value=0, ub=speed_limit, lb=0)
  m.x = m.CV(value=0)
  
  # variables that are adjusted (actuators)
  m.ac_ped = m.MV(value=0, lb=0, ub=100)
  m.br_ped = m.MV(value=0, lb=0, ub=50)
  
  # this should be changed to be an empirical (FOPDT type) relationship between v and ac_ped/br_ped
  
  # emperical equation values
  m.Ka = m.FV(value=1.58e-6)
  m.Da = m.FV(value=-326700)
  m.T  = m.FV(value=25.0)
  
  # set up the model variables. We will get measurements for these.
  m.eng_sp = m.SV(value=car.eng_wmin)
  m.eng_tq = m.SV(value=0)
  m.eng_tq1 = m.SV(value=0)
  m.eng_trq = m.SV(value=0)
  m.eng_br = m.SV(value=0)
  m.gear= m.SV(value=1)
  m.fuel= m.SV(value=0.8)
  m.fuel_int=m.SV(value=0)
      
  m.br_sign=m.sign2(m.br_ped)
  
  m.vchck1=m.Intermediate(m.v-0.1)
  m.vchck2=m.sign2(m.vchck1)
  m.vchck3=m.max2(m.vchck2,0)
  # set up the time variable
  m.Equation(m.t.dt() == 1)

  # set up the emperical equation
  m.Equation(m.T * m.v.dt() + m.v == m.Ka * (m.ac_ped - m.br_ped*m.vchck3) * (m.t - m.Da))
  m.eng_sp=m.max2(car.Fdr*m.gear*60/(2*np.pi)*m.v/car.wh_rd,car.eng_wmin)
  
  # set up the general equations
  m.Equation(m.v.dt() == m.a)
  m.Equation(m.x.dt() == m.v)
  
  m.cspline(m.eng_sp,m.eng_tq,car.eng_spd,car.eng_trq,True)
  m.cspline(m.eng_sp,m.eng_br,car.eng_spd,car.eng_brk,True)
  m.cspline(m.v,m.gear,car.v_new,car.gb_new,True)
  m.Equation(m.eng_tq1==m.eng_tq*m.ac_ped/100+m.eng_br*m.br_sign*m.vchck3)
  m.eng_trq=m.max2(m.eng_tq1,0)
  
  m.bspline(m.eng_sp,m.eng_trq,m.fuel,car.tck[0],car.tck[1],car.tck[2],data=False)
  m.Equation(m.fuel_int.dt()==m.fuel)
  
  
#%%
# Configure MHE
# 3 sec time horizon, steps of 0.1 sec
mhe.time = np.linspace(0, 3, 31)

# measured inputs
for s in (mhe.ac_ped, mhe.br_ped, mhe.v):#, mhe.whl_sp, mhe.whl_tq,mhe.a,):
  s.FSTATUS = 1 # receive measurements
  s.STATUS = 0 
    
mhe.v.MEAS_GAP=1.

mhe.x.STATUS=0
mhe.x.FSTATUS=0

# solved variables
for s in (mhe.Ka, mhe.Da, mhe.T):
  s.STATUS = 0 # turned on after 15 seconds
  s.FSTATUS = 0

mhe.options.IMODE = 5 # MHE
mhe.options.SOLVER = 1 #APOPT
mhe.options.CV_TYPE=1 #l1 norm
#%%
# Configure MPC
# tf sec time horizon, steps of tf/30 sec
mpc.time = np.linspace(0, 3, 31)
mpc.final=mpc.Param(final)
mpc.Equation(mpc.ac_ped*mpc.br_ped==0)

# measured inputs
for s in (mpc.Ka, mpc.Da, mpc.T):
  s.FSTATUS = 1
  s.STATUS = 0

mpc.v.STATUS=0
mpc.v.FSTATUS=0

mpc.x.STATUS=1
mpc.x.FSTATUS=1
mpc.x.SPLO=xgoals[0]-xldb
mpc.x.SPHI=xgoals[0]+xhdb
mpc.x.WSPLO=100
mpc.x.WSPHI=1e4

# adjusted parameters
for s in (mpc.ac_ped, mpc.br_ped):
  s.STATUS = 1
  s.FSTATUS = 0

#mpc.Obj(mpc.fuel_int*mpc.final)#+100000*((mpc.x-mpc_xgoal)*mpc.final)**2)

mpc.options.IMODE = 6 # MPC
mpc.options.SOLVER = 1 #APOPT
mpc.options.CV_TYPE=1 #l1 norm





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
gear      = [1,1]
i=0

grades=np.zeros(nsteps)

sp_store=[0.2]*2
vs=[0]*2
rand_vs=[0]*2


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


# for testing, make a schedule of ac_ped and br_ped



#variables assign
vs=[0]*2   #variable for actual vehicle speed
acc=[0]*2
wh_sp=[0]*2
wh_spt=[0]*2
gb_op=[0]*2
gb_opt=[0]*2
gb_ip=[0]*2
gb_ipt=[0]*2
gb_rat=[0]*2
gb_eff=[0]*2
eng_sp=[0]*2
eng_tq=[0]*2

eng_re=[0]*2



act_ped =[0]*2

v0        = 0.0
eng_wmin  = 1000
eng_wmax  = 9000
#eng_w     = eng_wmin
eng_t     = 0
vgear     = 0.0
#ws        = 0.0

# prediction values
v_pred =[0]*2
et_pred =[0]*2
es_pred =[0]*2
br_pred=[0]*2
ff_pred=[min(car.fuel_flow)]*2

time =[0]*2
sp_store=[0]*2
splo=[xgoals[0]-xldb]*2
sphi=[xgoals[0]+xhdb]*2

Ka_est=[1.58e-6]*2
Da_est =[-326700]*2
T_est  =[25.0]*2
x_est=[0]*2
ac_ped =[0]*2
br_ped =[0]*2

stop=0
# create a plot
plt.close('all')
plt.figure(figsize=(10, 7))
plt.ion()
plt.show()
for i in range(nsteps-1):
  time.append(time[-1]+delta_t)
  if vs[-1]<0.1 and x_est[-1]>xgoals[stop]:
    stop+=1
    
  mpc.x.SPLO=xgoals[stop]-xldb
  mpc.x.SPHI=xgoals[stop]+xhdb
  
  splo.append(mpc.x.SPLO)
  sphi.append(mpc.x.SPHI)
  
#  print(mpc.x.SPHI,mpc.x.SPLO)
  if i == 0/car.delta_t:
    sp = 25
#  if i == 50/car.delta_t:
#      sp = 0
#  if i == 100/car.delta_t:
#      sp = 15
#  if i == 150/car.delta_t:
#      sp = 20
#  if i == 200/car.delta_t:
#      sp = 10
#  if i == 250/car.delta_t:
#      sp = 0
  sp_store.append(sp)
  mhe.v.UPPER=sp
  mpc.v.UPPER=sp
  
  mpc.Ka.MEAS=Ka_est[-1]
  mpc.Da.MEAS=Da_est[-1]
  mpc.T.MEAS=T_est[-1]
  mpc.x.MEAS=x_est[-1]
  
  try:
      mpc.solve(disp=False)
      ac_ped.append(mpc.ac_ped.NEWVAL)
      br_ped.append(mpc.br_ped.NEWVAL)
      
  except:
    ac_ped.append(ac_ped[-1])
    br_ped.append(br_ped[-1])
    
    print("MPC Failed:{0}".format(datetime.now()))
  u=ac_ped[-1]-br_ped[-1]
  step.append(u)
  
  grade.append(grades[i])

  
  new_pts=vehicle(u,eng_w[-1],vs[-1],ws[-2:],delta_t,gb_op[-2:],grade[-1]) #[eng_tq, #0
#                                                                        gb_rat, #1
#                                                                        gb_eff, #2
#                                                                        gear,   #3
#                                                                        gb_opt, #4
#                                                                        wh_spt, #5
#                                                                        ws,     #6
#                                                                        gb_op,  #7
#                                                                        gb_ip,  #8
#                                                                        eng_w,  #9
#                                                                        vs,     #10
#                                                                        acc]    #11
  eng_tq.append(new_pts[0])
  gb_rat.append(new_pts[1])
  gb_eff.append(new_pts[2])
  gear.append(new_pts[3])
  gb_opt.append(new_pts[4])
  wh_spt.append(new_pts[5])
  ws.append(new_pts[6])
  gb_op.append(new_pts[7])
  gb_ip.append(new_pts[8])
  eng_w.append(new_pts[9])
  vs.append(new_pts[10])
  acc.append(new_pts[11])
  x_est.append(x_est[-1]+vs[-1]*car.delta_t)
  
  if i ==  15:
    for s in (mhe.Ka, mhe.Da, mhe.T):
        s.STATUS = 1
  
  # insert the measurements
  mhe.ac_ped.MEAS = ac_ped[-1]
  mhe.br_ped.MEAS = br_ped[-1]
  mhe.v.MEAS = vs[-1]

  
  # solve
  try:
    mhe.solve(disp=False)
    # update value
    Ka_est.append(mhe.Ka.NEWVAL)
    Da_est.append(mhe.Da.NEWVAL)
    T_est.append(mhe.T.NEWVAL)
    v_pred.append(mhe.v.MODEL)
    et_pred.append(mhe.eng_trq[-1])
    es_pred.append(mhe.eng_sp[-1])
#    br_pred.append(mhe.eng_br[-1])
    ff_pred.append(mhe.fuel[-1])
#    x_est.append(mhe.x[-1])
  except KeyboardInterrupt :
    print('stopping')
    break
  except:
    Ka_est.append(Ka_est[-1])
    Da_est.append(Da_est[-1])
    T_est.append(T_est[-1])
    v_pred.append(v_pred[-1])
    et_pred.append(et_pred[-1])
    es_pred.append(es_pred[-1])
    br_pred.append(br_pred[-1])
    ff_pred.append(ff_pred[-1])
#    x_est.append(x_est[-1])
    print("failed!")
  # plug A, B, C, D, E, and F into the mpc and solve for the next pedal values
  # plot progress
  plt.clf()
  plt.subplot(511)
  plt.plot(time, vs, 'b.', label='Car')
  plt.plot(time, v_pred, 'r-', label='Model')
  plt.ylabel('Velocity')
  plt.xlabel('time')
  plt.legend()
  plt.subplot(512)
  plt.plot(time, x_est, 'r-', label='Position')
  plt.ylabel('Position (m)')

  plt.plot(time, sphi, 'k--', label='SP')
  plt.plot(time, splo, 'k--', label='')
  plt.xlabel('time')
  plt.legend()
  plt.subplot(514)
  plt.plot(time,ac_ped,label='Acceleration')
  plt.plot(time,br_ped,label='Braking')
  plt.ylabel('% pressed')
  plt.xlabel('Time')
  plt.legend(loc=2)
#  plt.subplot(513)
#  plt.plot(time, eng_tq, 'b.', label='Car')
#  plt.plot(time, et_pred, 'r-', label='Model')
##  plt.plot(time, br_pred, 'g-', label='Model u<=0')
#  plt.ylabel('Engine Torque (Nm)')
#  plt.xlabel('time')
#  plt.legend()
#  plt.subplot(514)
#  plt.plot(time, eng_w, 'b.', label='Car')
#  plt.plot(time, es_pred, 'r-', label='Model')
#  plt.ylabel('Engine Speed (rpm)')
#  plt.xlabel('Time')
#  plt.legend()
  plt.subplot(515)
  
  plt.plot(time, ff_pred, 'r-', label='Model')
  plt.ylabel('Fuel flow (L/min)')
  plt.xlabel('Time')
  plt.legend()
  
  plt.draw()
  plt.pause(0.001)
  
  # loop
  #%%
plt.figure()
plt.subplot(511)
plt.plot(time, vs, 'b.', label='Car')
plt.plot(time, v_pred, 'r-', label='Model')
plt.ylabel('Velocity')
plt.xlabel('time')
plt.legend()
plt.subplot(512)
plt.plot(time, gb_rat, 'b.', label='Gear')
plt.plot(time, np.array(vs)/np.array(v_pred), 'g:', label='%err in v')
plt.xlabel('time')
plt.legend()
plt.subplot(513)
plt.plot(time, np.array(eng_tq), 'b.', label='Car')
plt.plot(time, np.array(et_pred)*np.array(ac_ped)/100, 'r-', label='Model u>0')
#plt.plot(time, br_pred, 'g-', label='Model u<=0')
plt.ylabel('Engine Torque (Nm)')
plt.xlabel('time')
plt.legend()
plt.subplot(514)
plt.plot(time, eng_w, 'b.', label='Car')
plt.plot(time, es_pred, 'r-', label='Model')
plt.ylabel('Engine Speed (rpm)')
plt.xlabel('Time')
plt.legend()
plt.subplot(515)

plt.plot(time, ff_pred, 'r-', label='Model')
plt.ylabel('Fuel flow (L/min)')
plt.xlabel('Time')
plt.legend()
plt.show()