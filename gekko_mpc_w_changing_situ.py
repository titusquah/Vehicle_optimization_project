import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import odeint
from gekko import GEKKO
import car_v03 as car

#%%Parameters
num_points = 50 # more points is more accurate, but every point adds 2 DOF
max_time = 500
# set up path
x_goal = 5000 # m
speed_limit = 25 #m/s
stop_sign = "no"

gcv = [0.2, 7.1511, 11.1736, 17.8778, 20.5594] # m/s

#%%Vehicle simulator
def vehicle(u,eng_w,v0,ws,delta_t):
  if u > 0:
    eng_tq= car.eng_wot(eng_w) * u/100
  else:   
    eng_tq= car.eng_wot(eng_w)
  gb_rat, gb_eff = car.g_box(v0)
  gear= car.gb[int(gb_rat)]
  gb_opt=eng_tq * gear * gb_eff
  wh_spt= gb_opt * car.Fdr * car.Fef
  wh_spd= odeint(car.roadload, ws, [0,delta_t], args=(wh_spt,u))
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

#%%Gekko MPC
mpc=GEKKO()
mpc.time=np.linspace(0,1,num_points) 

# set up the variables
x = mpc.SV(value=0, lb=0, ub=x_goal)
v = mpc.SV(value=0, lb=0, ub=speed_limit) #vs
a = mpc.SV(value=0, ub=5, lb=-5)          #acc

# set up the Manipulated Variables
ac_ped = mpc.MV(value=0, lb=0, ub=100) #+u
br_ped = mpc.MV(value=0, lb=0, ub=100) #-u

# set up the gears (1 is in the gear, 0 is not in the gear)
in_gr = [mpc.MV(integer=True, value=0, lb=0, ub=1) for i in range(5)] #gvar
in_gr[0].VALUE=1

gear_ratio = mpc.SV(value=car.ge[0])

# set up the time variable (to minimize)
tf = mpc.CV(value=100, lb=1, ub=max_time) #s

# turn them 'on'
for s in (ac_ped, br_ped, tf,in_gr_1, in_gr_2, in_gr_3):
	s.STATUS = 1
    




# turn on the gears
for s in ():
	s.STATUS = 1
# set the cutoffs for the gears


# I'm going to assume that car's fuel pump is a plunger type,
# so the fuel rate is proportional to the %pedal
# This means I can minimize fuel usage by minimizing the ac_ped variable

# add stops
mpc.fix(x, num_points-1, x_goal) # destination

if (stop_sign == "yes"):
	# stop sign
	mpc.fix(v, int(num_points/3), 0) 

# set up the governing equations for the car
mpc.Equation(x.dt() / tf == v)
mpc.Equation(v.dt() / tf == a)
mpc.Equation(a == 1.0/(car.m+car.load) * \
				(car.Fp*ac_ped*gear_ratio*car.Fdr - \
				0.5*car.rho*car.Cd*car.A*v**2 - \
				car.Crr*(car.m+car.load)) - \
				1.0 / (car.m+car.load) * \
				(car.Fb*br_ped - \
				0.5*car.rho*car.Cd*car.A*v**2))

				
# don't use break and accelerator at the same time
mpc.Equation(ac_ped * br_ped == 0)

# set up the gear logic (pick 1 of 4)
mpc.Equation(in_gr_1 * (v-gcv[0]) <= 0)# in gear 1 when v < gcv[0]
mpc.Equation(in_gr_2 * (v-gcv[0]) * (v-gcv[1]) <= 0) # in gear 2 when v < gcv[1]
mpc.Equation(in_gr_3 * (v-gcv[1]) >= 0) # in gear 3 when v > gcv[2]
mpc.Equation(in_gr_1 + in_gr_2 + in_gr_3 == 1) # must be in a gear

# set the gear ratio based on the gear
mpc.Equation(gear_ratio == car.ge[0]*in_gr_1 + car.ge[1]*in_gr_2 + car.ge[2]*in_gr_3)

# set up the objective
last = np.zeros(num_points)
last[-1] = 1
last = mpc.Param(value=last)

# set up the solver
mpc.options.IMODE = 6
mpc.options.SOLVER = 1


# set up the objective
mpc.Obj(100*ac_ped + tf)# + 1000*(1-in_gr_1-in_gr_2-in_gr_3))
#%% Simulate
test_data=[[0]]*6 #[x,v,a,gear,acc,bra]


