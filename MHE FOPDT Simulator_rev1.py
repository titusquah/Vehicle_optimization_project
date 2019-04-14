import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import car_v04 as car # this is vehicle_model_v04.py in git
from scipy.integrate import odeint
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
speed_limit=30
# set up the gekko models
mhe = GEKKO(name='mhe')
mpc = GEKKO(name='mpc')

# set up the empirical equations relating pedal% to velocity
for m in (mhe, mpc):
  m.tf = m.FV(value=3, lb=0)
  m.tf.FSTATUS=0
  m.tf.STATUS=0
  
  # time
  m.t = m.Var()
  
  # variables to keep track of
  m.a = m.SV(value=0)
  m.v = m.CV(value=0, ub=speed_limit, lb=0)
  m.x = m.SV(value=0)
  
  # variables that are adjusted (actuators)
  m.ac_ped = m.MV(value=0, lb=0, ub=100)
  m.br_ped = m.MV(value=0, lb=0, ub=100)
  
  # this should be changed to be an empirical (FOPDT type) relationship between v and ac_ped/br_ped
  
  # emperical equation values
  m.Ka = m.FV(value=1.58e-6)
  m.Da = m.FV(value=-326700)
  m.T  = m.FV(value=25.0)
  m.G  = m.MV(value=1.0, ub=7, lb=0)
  
  # set up the model variables. We will get measurements for these.
  m.eng_sp = m.SV(value=car.eng_wmin)
  m.eng_tq = m.SV(value=0)
  m.eng_tq1 = m.SV(value=0)
  m.eng_trq = m.SV(value=0)
  m.eng_br = m.SV(value=0)
  m.gear= m.SV(value=1)
  m.fuel= m.SV(value=0.8)
      
  m.br_sign=m.sign2(m.br_ped)
  # set up the time variable
  m.Equation(m.t.dt()/m.tf == 1)

  # set up the emperical equation
  m.Equation(m.T * m.v.dt()/m.tf + m.v == m.Ka * (m.ac_ped - m.br_ped) * (m.t - m.Da))
  m.eng_sp=m.max2(car.Fdr*m.gear*60/(2*np.pi)*m.v/car.wh_rd,car.eng_wmin)
  
  # set up the general equations
  m.Equation(m.v.dt()/m.tf == m.a)
  m.Equation(m.x.dt()/m.tf == m.v)
  
  m.cspline(m.eng_sp,m.eng_tq,car.eng_spd,car.eng_trq,True)
  m.cspline(m.eng_sp,m.eng_br,car.eng_spd,car.eng_brk,True)
  m.cspline(m.v,m.gear,car.v_new,car.gb_new,True)
  m.Equation(m.eng_tq1==m.eng_tq*m.ac_ped/100+m.eng_br*m.br_sign)
  m.eng_trq=m.max2(m.eng_tq1,0)
  
  m.bspline(m.eng_sp,m.eng_trq,m.fuel,car.tck[0],car.tck[1],car.tck[2],data=False)
  
  
	
# Configure MHE
# 3 sec time horizon, steps of 0.1 sec
mhe.time = np.linspace(0, 1, 31)

# measured inputs
for s in (mhe.ac_ped, mhe.br_ped, mhe.v):#, mhe.whl_sp, mhe.whl_tq,mhe.a,):
	s.FSTATUS = 1 # receive measurements
    
mhe.v.STATUS=1
mhe.v.MEAS_GAP=1.
	
# solved variables
for s in (mhe.Ka, mhe.Da, mhe.T):
	s.STATUS = 0 # turned on after 15 seconds
	
mhe.options.IMODE = 5 # MHE
mhe.options.SOLVER = 1 #APOPT
mhe.options.CV_TYPE=1 #l1 norm

# Configure MPC
# tf sec time horizon, steps of tf/30 sec
mpc.time = np.linspace(0, 1, 31)

# measured inputs
for s in (mpc.Ka, mpc.Da, mpc.T):
	s.FSTATUS = 1
	
# adjusted parameters
for s in (mpc.ac_ped, mpc.br_ped):
	s.STATUS = 1
	
# set point is final position
#mpc.x.SPHI = x_goal
#mpc.x.SPLO = x_goal

mpc.options.IMODE = 6 # MPC

# create a plot
plt.close('all')
plt.figure(figsize=(10, 7))
plt.ion()
plt.show()


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


# for testing, make a schedule of ac_ped and br_ped
ac_ped =[0]*2
br_ped =[0]*2


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
es=[0]*2
ies=[0]*2
sp_store=[0]*2
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

for i in range(nsteps-1):
  time.append(time[-1]+delta_t)
  if i == 0.5/car.delta_t:
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

  if vs[-1] < 0.1 and u < 0:
      act_ped.append(-50)
  else:
      act_ped.append(u)
  grade.append(grades[i])
  if u>=0:
    ac_ped.append(u)
    br_ped.append(0)
  else:
    ac_ped.append(0)
    br_ped.append(-u)
  
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
  if i ==  15:
    for s in (mhe.Ka, mhe.Da, mhe.T):
        s.STATUS = 1
  		
  # insert the measurements
  mhe.ac_ped.MEAS = ac_ped[-1]
  mhe.br_ped.MEAS = br_ped[-1]
  
  
  
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
  
#  mhe.a.MEAS = acc[-1]
  mhe.v.MEAS = vs[-1]
#  mhe.whl_sp.MEAS = ws[-1]
#  mhe.whl_tq.MEAS = eng_tq[-1]
  
  # solve
  try:
    mhe.solve(disp=False)
    # update value
    v_pred.append(mhe.v.MODEL)
    et_pred.append(mhe.eng_trq[-1])
    es_pred.append(mhe.eng_sp[-1])
#    br_pred.append(mhe.eng_br[-1])
    ff_pred.append(mhe.fuel[-1])
  except KeyboardInterrupt :
    print('stopping')
    break
  except:
    v_pred.append(v_pred[-1])
    et_pred.append(et_pred[-1])
    es_pred.append(es_pred[-1])
    br_pred.append(br_pred[-1])
    ff_pred.append(ff_pred[-1])
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
  plt.plot(time, gb_rat, 'b.', label='Gear')
  plt.plot(time, np.array(vs)/np.array(v_pred), 'g:', label='%err in v')
  plt.xlabel('time')
  plt.legend()
  plt.subplot(513)
  plt.plot(time, eng_tq, 'b.', label='Car')
  plt.plot(time, et_pred, 'r-', label='Model')
#  plt.plot(time, br_pred, 'g-', label='Model u<=0')
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