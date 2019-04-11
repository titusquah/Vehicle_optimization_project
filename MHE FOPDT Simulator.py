import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
import NewCar as car # this is vehicle_model_v04.py in git
from scipy.integrate import odeint

# set up path
x_goal = 1000 # m
speed_limit = 30 #m/s
stop_sign = "nope"
max_time = 300
# set up the gekko models
mhe = GEKKO(name='mhe')
mpc = GEKKO(name='mpc')

# set up the empirical equations relating pedal% to velocity
for m in (mhe, mpc):
	# time
	m.t = m.Var()
	
	# variables to keep track of
	m.a = m.CV(value=0, ub=2, lb=-2)
	m.v = m.CV(value=0, ub=speed_limit, lb=0)
	m.x = m.CV(value=0)
	
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
	m.whl_sp = m.CV(value=0)
	m.whl_tq = m.CV(value=0)
	
	# set up the time variable
	m.Equation(m.t.dt() == 1)

	# set up the emperical equation
	m.Equation(m.T * m.v.dt() + m.v == m.Ka * (m.ac_ped - m.br_ped) * (m.t - m.Da))

	# set up the governing equations for the model
	m.whl_I = m.Const(((car.m + car.load)*car.wh_rd**2) + car.wh_inf + car.wh_inr)	
	m.Equation(m.whl_sp.dt() == 1/m.whl_I * (m.whl_tq \
						- 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*m.whl_sp**2 \
						- car.wh_rd*car.Crr*(car.m+car.load)*m.whl_sp) +\
						1/m.whl_I * (car.Fb*m.br_ped*car.wh_rd \
						- 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*m.whl_sp**2 \
						- car.wh_rd*car.Crr*(car.m+car.load)*m.whl_sp))
	m.Equation(m.v == m.whl_sp * car.wh_rd)
	
	# set up the general equations
	m.Equation(m.v.dt() == m.a)
	m.Equation(m.x.dt() == m.v)
	
# Configure MHE
# 30 sec time horizon, steps of 1 sec
mhe.time = np.linspace(0, 30, 31)

# measured inputs
for s in (mhe.ac_ped, mhe.br_ped, mhe.a, mhe.v, mhe.whl_sp, mhe.whl_tq):
	s.FSTATUS = 1 # receive measurements
	
# solved variables
for s in (mhe.Ka, mhe.Da, mhe.T):
	s.STATUS = 0 # turned on after 15 seconds
	
mhe.options.IMODE = 5 # MHE

# Configure MPC
# 30 sec time horizon, steps of 1 sec
mpc.time = np.linspace(0, 30, 31)

# measured inputs
for s in (mpc.Ka, mpc.Da, mpc.T):
	s.FSTATUS = 1
	
# adjusted parameters
for s in (mpc.ac_ped, mpc.br_ped):
	s.STATUS = 1
	
# set point is final position
mpc.x.SPHI = x_goal
mpc.x.SPLO = x_goal

mpc.options.IMODE = 6 # MPC

# create a plot
plt.figure(figsize=(10, 7))
plt.ion()
plt.show()

done = False
i = 0

# for testing, make a schedule of ac_ped and br_ped
ac_ped = np.zeros(max_time+1)
br_ped = np.zeros(max_time+1)
ac_ped[10:] = 40
ac_ped[100:] = 100
ac_ped[200:] = 0
br_ped[200:] = 90
br_ped[250:] = 50

#variables assign
vs        = np.zeros(max_time+1)   #variable for actual vehicle speed
acc       = np.zeros(max_time+1)
wh_sp     = np.zeros(max_time+1)
wh_spt    = np.zeros(max_time+1)
gb_op     = np.zeros(max_time+1)
gb_opt    = np.zeros(max_time+1)
gb_ip     = np.zeros(max_time+1)
gb_ipt    = np.zeros(max_time+1)
gb_rat    = np.zeros(max_time+1)
gb_eff    = np.zeros(max_time+1)
eng_sp    = np.zeros(max_time+1)
eng_tq    = np.zeros(max_time+1)
eng_re    = np.zeros(max_time+1)
es        = np.zeros(max_time+1)
ies       = np.zeros(max_time+1)
sp_store  = np.zeros(max_time+1)
act_ped   = np.zeros(max_time+1)

v0        = 0.0
eng_wmin  = 1000
eng_wmax  = 9000
eng_w     = eng_wmin
eng_t     = 0
vgear     = 0.0
ws        = 0.0

delta_t = 1

# prediction values
v_pred = np.zeros(max_time+1)
wt_pred = np.zeros(max_time+1)
time = np.linspace(0, max_time, max_time+1)

for i in range(max_time):

	# go through the car equations to get the current speed based on the pedals
	# start by calculating 'u' for the equations
	u = ac_ped[i+1]
	if ac_ped[i+1] == 0:
		u = -br_ped[i+1]

	if v0 < 0.1 and u < 0:
		act_ped[i+1] = -50
	else:
		act_ped[i+1] = u
		
	if u > 0:
		eng_tq[i+1]= car.eng_wot(eng_w, u) * act_ped[i+1]/100
	if u <= 0:
		eng_tq[i+1]= car.eng_wot(eng_w, u)

	gb_rat[i+1], gb_eff[i+1] = car.g_box(vgear)
	gear       = car.gb[int(gb_rat[i+1])]
	gb_ipt[i+1]  = eng_tq[i+1] #- Igb_i * (gb_ip[i] - gb_ip[i-1])/delta_t #+ vdamp * gb_ip[i] #** 2 #+ mdamp * gb_ip[i]
	gb_opt[i+1]  = gb_ipt[i+1] * gear * gb_eff[i+1] - (car.Igb_o + car.Igb_i) * (gb_op[i] - gb_op[i-1])/delta_t
	
	wh_spt[i+1]  = gb_opt[i+1] * car.Fdr * car.Fef - (car.wh_inf + car.wh_inr) * (wh_sp[i] - wh_sp[i-1])/delta_t
	whl_t      = wh_spt[i+1]
	wh_spd     = odeint(car.roadload, ws, [0,delta_t], args=(whl_t,u))
	
	ws         = wh_spd[-1]
	wh_sp[i+1]   = ws
	
	gb_op[i+1]   = wh_sp[i+1] * car.Fdr
	gb_ip[i+1]   = gb_op[i+1] * gear
	eng_sp[i+1]  = gb_ip[i+1] * 60 / (2 * np.pi)
	
	if eng_sp[i+1] < eng_wmin:
		eng_sp[i+1] = eng_wmin
	if eng_sp[i+1] > eng_wmax:
		eng_sp[i+1] = eng_wmax
	eng_w      = eng_sp[i+1]
	vs[i+1]    = wh_sp[i+1] * car.wh_rd
	v0         = vs[i+1]
	vgear      = vs[i+1]
	acc[i] = (vs[i+1] - vs[i]) / (delta_t * 9.81)

	# run the mhe to get Ka, Kb, and T (after 10 sec)		
	if i == 15:
		for s in (mhe.Ka, mhe.Da, mhe.T):
			s.STATUS = 1
		
	# insert the measurements
	mhe.ac_ped.MEAS = ac_ped[i+1]
	mhe.br_ped.MEAS = br_ped[i+1]
	mhe.a.MEAS = acc[i]
	mhe.v.MEAS = vs[i+1]
	mhe.whl_sp.MEAS = ws
	mhe.whl_tq.MEAS = whl_t
	
	# solve
	try:
		mhe.solve(disp=False)
		# update value
		v_pred[i+1] = mhe.v.MODEL
		wt_pred[i+1] = mhe.whl_tq.MODEL
	except:
		v_pred[i+1] = v_pred[i]
		wt_pred[i+1] = wt_pred[i]
		print("failed!")

	# plug A, B, C, D, E, and F into the mpc and solve for the next pedal values

	# plot progress
	plt.clf()
	plt.subplot(411)
	plt.plot(time[:i+1], vs[:i+1], 'b.', label='Car')
	plt.plot(time[:i+1], v_pred[:i+1], 'r-', label='Model')
	plt.ylabel('Velocity')
	plt.xlabel('time')
	plt.legend()
	plt.subplot(412)
	plt.plot(time[:i+1], gb_rat[:i+1], 'b.', label='Gear'))
	plt.plot(time[:i+1], vs[:i+1]/v_pred[:i+1], 'g:', label='%err in v')
	plt.xlabel('time')
	plt.legend()
	plt.subplot(413)
	plt.plot(time[:i+1], wh_spt[:i+1], 'b.', label='Car')
	plt.plot(time[:i+1], wt_pred[:i+1], 'r-', label='Model')
	plt.ylabel('Wheel Torque')
	plt.xlabel('time')
	plt.legend()
	plt.subplot(414)
	plt.plot(wh_spt[:i+1], vs[:i+1], '.')
	plt.ylabel('Velocity')
	plt.xlabel('Wheel Torque')
	plt.pause(0.001)

	# loop
