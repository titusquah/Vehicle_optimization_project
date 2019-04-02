import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import odeint
from gekko import GEKKO
import vehicle_model_v03 as car

#%%Parameters
num_points = 50 # more points is more accurate, but every point adds 2 DOF
max_time = 500
# set up path
x_goal = 5000 # m
speed_limit = 25 #m/s
stop_sign = "no"



#%%Gekko MPC
mpc=GEKKO()
mpc.time=np.linspace(0,1,num_points) 

# set up the Manipulated Variables
ac_ped = mpc.MV(value=0, lb=0, ub=100)
br_ped = mpc.MV(value=0, lb=0, ub=100)

# set up the time variable (to minimize)
tf = mpc.CV(value=100, lb=1, ub=max_time)

# turn them 'on'
for s in (ac_ped, br_ped, tf):
	s.STATUS = 1
    
# set up the variables
x = mpc.SV(value=0, lb=0, ub=x_goal)
v = mpc.SV(value=0, lb=0, ub=speed_limit)
a = mpc.SV(value=0, ub=2, lb=-2)

# set up the gears (1 is in the gear, 0 is not in the gear)
in_gr_1 = mpc.MV(integer=True, value=1, lb=0, ub=1) 
in_gr_2 = mpc.MV(integer=True, value=0, lb=0, ub=1)
in_gr_3 = mpc.MV(integer=True, value=0, lb=0, ub=1)
gear_ratio = mpc.SV(value=car.ge[0])

# turn on the gears
for s in (in_gr_1, in_gr_2, in_gr_3):
	s.STATUS = 1
# set the cutoffs for the gears
gcv = [0.2, 7.1511, 11.1736, 17.8778, 20.5594] # m/s

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

# solve
mpc.solve()

# plot the results
time = np.linspace(0, 1, num_points)*tf.NEWVAL
plt.figure(figsize=(10, 10))
plt.subplot(511)
plt.plot(time, np.array(x.value))
plt.plot([0, tf.NEWVAL], np.full(2,x_goal), label='goal')
plt.ylabel('position\n(m)')
plt.subplot(512)
plt.plot(time, np.array(v.value))
plt.ylabel('velocity\n(m/s)')
plt.subplot(513)
plt.plot(time, np.array(a.value))
plt.ylabel('acceleration\n(m/s/s)')
plt.subplot(514)
plt.plot(time, np.array(in_gr_1.value)-0.2, 'x', label='Gear 1')
plt.plot(time, np.array(in_gr_2.value)-0.1, 'o', label='Gear 2')
plt.plot(time, in_gr_3, 'D', label='Gear 3')
plt.ylabel('Selected Gear\n ')
plt.ylim([0.5,1.5])
plt.legend(loc=2)
plt.subplot(515)
plt.plot(time, br_ped, label='Brake')
plt.plot(time, ac_ped, label='Accelerator')
plt.ylabel('Pedal Position\n (%)')
plt.xlabel('Time (s)')
plt.legend(loc=2)
plt.show()

# set up the plot
plt.figure(figsize=(10, 5))
plt.ion()

# add some animation because it's cool
xs = np.array(x.value)

if (stop_sign == "yes"):
	# get the stop sign position
	st_ps = xs[int(num_points/3)]

for i in range(len(mpc.time)):
	plt.clf()
	# the car can be a green box. the road will be a black line.
	# set the ylim
	plt.ylim([-0.1, 0.5])
	
	# start with the road
	plt.plot([0, x_goal], [0, 0], 'k-', linewidth=5)
	
	if (stop_sign == "yes"):
		# draw the stop sign
		plt.plot([st_ps, st_ps], [0, 0.1], 'k-', linewidth=2)
		plt.plot(st_ps, 0.1, 'rH', markersize=15)
	
	# now plot the car
	plt.plot(xs[i], 0.01, 'gs', markersize=20, label='Car')
	plt.legend()
	
	# plot it
	plt.legend(loc=2)
	plt.draw()
	plt.pause(0.1)