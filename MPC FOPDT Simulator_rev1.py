import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import car_v04 as car # this is vehicle_model_v04.py in git
from scipy.integrate import odeint

xgoals=[10.]
xdbhi=0.3
xdblo=0.7
final=np.zeros(31)
final[-1]=1
initial=np.zeros(31)
initial[0]=1
speed_limit=1
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
  ws=max([ws,0])
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
#%% MHE
mhe = GEKKO(name='mhe')
mhe.time = np.linspace(0, 3, 31)

mhe_v = mhe.CV(value=0, ub=speed_limit, lb=0)

mhe_Ka = mhe.FV(value=1.58e-6)
mhe_Da = mhe.FV(value=-326700)
mhe_T  = mhe.FV(value=25.0)

# solved variables
for s in (mhe_Ka, mhe_Da, mhe_T):
  s.STATUS = 0 # turned on after 15 seconds
  s.FSTATUS = 0

mhe_ac_ped = mhe.MV(value=0, lb=0, ub=100)
mhe_br_ped = mhe.MV(value=0, lb=0, ub=50)

for s in (mhe_ac_ped, mhe_br_ped, mhe_v):#, mhe_whl_sp, mhe_whl_tq,mhe_a,):
  s.FSTATUS = 1 # receive measurements
  s.STATUS=0
  
mhe_v.MEAS_GAP=1.
	
#mhe_a = mhe.SV(value=0)
mhe_x=mhe.SV(value=0)
mhe_t = mhe.SV(0)
mhe_eng_sp = mhe.SV(value=car.eng_wmin)
mhe_eng_tq = mhe.SV(value=0)
mhe_eng_tq1 = mhe.SV(value=0)
mhe_eng_trq = mhe.SV(value=0)
mhe_eng_br = mhe.SV(value=0)
mhe_gear= mhe.SV(value=1)
mhe_fuel= mhe.SV(value=0.8)
#mhe_fuel_int=mhe.SV(value=0)

mhe_br_sign=mhe.sign2(mhe_br_ped)
mhe_vtest=mhe.sign2((mhe_v-0.1))
mhe_vtest2=mhe.max2(mhe_vtest,0)

mhe.cspline(mhe_v,mhe_gear,car.v_new,car.gb_new,True)
mhe_eng_sp=mhe.max2(car.Fdr*mhe_gear*60/(2*np.pi)*mhe_v/car.wh_rd,car.eng_wmin)

mhe.cspline(mhe_eng_sp,mhe_eng_tq,car.eng_spd,car.eng_trq,True)
mhe.cspline(mhe_eng_sp,mhe_eng_br,car.eng_spd,car.eng_brk,True)

mhe.Equation(mhe_t.dt() == 1)
mhe.Equation(mhe_T * mhe_v.dt() + mhe_v == mhe_Ka * (mhe_ac_ped - mhe_br_ped*mhe_vtest2) * (mhe_t - mhe_Da))
#mhe.Equation(mhe_v.dt() == mhe_a)
mhe.Equation(mhe_x.dt() == mhe_v)

mhe.Equation(mhe_eng_tq1==mhe_eng_tq*mhe_ac_ped/100+mhe_eng_br*mhe_br_sign)

mhe_eng_trq=mhe.max2(mhe_eng_tq1,0)

mhe.bspline(mhe_eng_sp,mhe_eng_trq,mhe_fuel,car.tck[0],car.tck[1],car.tck[2],data=False)

#mhe.Equation(mhe_fuel_int.dt()==mhe_fuel)

mhe.options.IMODE = 5 # MHE
mhe.options.SOLVER = 1 #APOPT
mhe.options.CV_TYPE=1 #l1 norm

#%%MPC
mpc = GEKKO(name='mpc')
mpc.time = np.linspace(0, 3, 31)

mpc_final=mpc.Param(final)

mpc_x=mpc.CV(value=0)

mpc_x.SPHI=xgoals[0]+xdbhi
mpc_x.SPLO=xgoals[0]-xdblo
mpc_x.TR_INIT=0
mpc_x.WSPHI=1e5
mpc_x.WSPLO=10
mpc_x.STATUS=1
mpc_x.FSTATUS=1

# emperical equation values
mpc_Ka = mpc.FV(value=1.58e-6)
mpc_Da = mpc.FV(value=-326700)
mpc_T  = mpc.FV(value=25.0)

for s in (mpc_Ka, mpc_Da, mpc_T):
  s.STATUS = 0
  s.FSTATUS = 1

# variables that are adjusted (actuators)
mpc_ac_ped = mpc.MV(value=0, lb=0, ub=100)
mpc_br_ped = mpc.MV(value=0, lb=0, ub=50)

for s in (mpc_ac_ped, mpc_br_ped):
  s.FSTATUS = 0 
  s.STATUS=1
  
mpc_ac_ped.DMAX=50
mpc_br_ped.DMAX=45

mpc_t = mpc.SV(0)
#mpc_a = mpc.SV(value=0)
mpc_v = mpc.SV(value=0, ub=speed_limit, lb=0)
mpc_eng_sp = mpc.SV(value=car.eng_wmin)
mpc_eng_tq = mpc.SV(value=0)
mpc_eng_tq1 = mpc.SV(value=0)
mpc_eng_trq = mpc.SV(value=0)
mpc_eng_br = mpc.SV(value=0)
mpc_gear= mpc.SV(value=1)
mpc_fuel= mpc.SV(value=0.8)
mpc_fuel_int=mpc.SV(value=0)

mpc_br_sign=mpc.sign2(mpc_br_ped)
mpc_vtest=mpc.sign2((mpc_v-0.1))
mpc_vtest2=mpc.max2(mpc_vtest,0)

mpc.cspline(mpc_v,mpc_gear,car.v_new,car.gb_new,True)
mpc_eng_sp=mpc.max2(car.Fdr*mpc_gear*60/(2*np.pi)*mpc_v/car.wh_rd,car.eng_wmin)

mpc.cspline(mpc_eng_sp,mpc_eng_tq,car.eng_spd,car.eng_trq,True)
mpc.cspline(mpc_eng_sp,mpc_eng_br,car.eng_spd,car.eng_brk,True)

mpc.Equation(mpc_t.dt() == 1)
mpc.Equation(mpc_T * mpc_v.dt() + mpc_v == mpc_Ka * (mpc_ac_ped - mpc_br_ped*mpc_vtest2) * (mpc_t - mpc_Da))
mpc.Equation(mpc_ac_ped*mpc_br_ped==0)
#mpc.Equation(mpc_v.dt() == mpc_a)
mpc.Equation(mpc_x.dt() == mpc_v)

mpc.Equation(mpc_eng_tq1==mpc_eng_tq*mpc_ac_ped/100+mpc_eng_br*mpc_br_sign)

mpc_eng_trq=mpc.max2(mpc_eng_tq1,0)

mpc.bspline(mpc_eng_sp,mpc_eng_trq,mpc_fuel,car.tck[0],car.tck[1],car.tck[2],data=False)

mpc.Equation(mpc_fuel_int.dt()==mpc_fuel)

#mpc.Obj(mpc_fuel_int*mpc_final)#+100000*((mpc_x-mpc_xgoal)*mpc_final)**2)

mpc.options.IMODE = 6 # MPC
mpc.options.CV_TYPE=1
mpc.options.SOLVER=1

#%%Parameters
#Simulation time step definition
tf        = 300                 #final time for simulation
nsteps    = 3001                 #number of time steps
delta_t   = tf / (nsteps - 1)   #length of each time step
#Advanced cyber driver
step      = [0]   #assigning array for pedal position
sp        = 0
gear      = [1,1]

grades=np.zeros(nsteps)

eng_w = [car.eng_wmin]*2
ws=[0]*2
grade=[0]*2
time=[0,0]

Ka_est=[1.58e-6]*2
Da_est=[-326700]*2
T_est=[25]*2

x_est=[0]*2
tf_est=[0]*2

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


sp_store=[0]*2
act_ped =[0]*2

# prediction values
v_pred =[0]*2
et_pred =[0]*2
es_pred =[0]*2
br_pred=[0]*2
ff_pred=[min(car.fuel_flow)]*2

time =[0]*2


# create a plot
plt.close('all')
plt.figure(figsize=(10, 7))
plt.ion()
plt.show()
make_mp4 = False

for i in range(nsteps-1):
  try:
    time.append(time[-1]+delta_t)
    if i == 0.5/car.delta_t:
      sp = 25
  #  if i == 5/car.delta_t:
  #      sp = 20
  #  if i == 10/car.delta_t:
  #      sp = 15
  #  if i == 15/car.delta_t:
  #      sp = 20
  #  if i == 20/car.delta_t:
  #      sp = 10
  #  if i == 25/car.delta_t:
  #      sp = 0
    sp_store.append(sp)
    
    mpc_v.UPPER=sp
    mhe_v.upper=sp
    
    mpc_x.MEAS=x_est[-1]
    mpc_Ka.MEAS=Ka_est[-1]
    mpc_Da.MEAS=Da_est[-1]
    mpc_T.MEAS=T_est[-1]
    try:
      mpc.solve(disp=False)
      ac_ped.append(mpc_ac_ped.NEWVAL)
      br_ped.append(mpc_br_ped.NEWVAL)
      
    except:
      ac_ped.append(0)
      br_ped.append(10)
      
      
      print("MPC Failed")
      
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
    
    if i ==  15:
      for s in (mhe_Ka, mhe_Da, mhe_T):
          s.STATUS = 1
#    print(2000+i)
    		
    # insert the measurements
    mhe_ac_ped.MEAS = ac_ped[-1]
    mhe_br_ped.MEAS = br_ped[-1]
  
    mhe_v.MEAS = vs[-1]
  
    
    # solve
    try:
      mhe.solve(disp=False)
      # update value
      v_pred.append(mhe_v.MODEL)
      et_pred.append(mhe_eng_tq1[-1])
      es_pred.append(mhe_eng_sp[-1])
  #    br_pred.append(mhe_eng_br[-1])
      ff_pred.append(mhe_fuel[-1])
      x_est.append(mhe_x[-1])
    except:
      v_pred.append(v_pred[-1])
      et_pred.append(et_pred[-1])
      es_pred.append(es_pred[-1])
      br_pred.append(br_pred[-1])
      ff_pred.append(ff_pred[-1])
      x_est.append(x_est[-1])
      print("failed!")
#    print(3000+i)
    
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
#    print(4000+i)
  except KeyboardInterrupt:
    print("stopping")
    break
  
