import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from gekko import GEKKO
import car_v04 as car

#%%Parameters
num_points = 50 # more points is more accurate, but every point adds 2 DOF
max_time = 500
# set up path
x_goal = 5000 # m
speed_limit = 25 #m/s
stop_sign = "no"

gcv = [7.1511, 11.1736, 17.8778, 20.5594] # m/s

#%%Vehicle simulator
def vehicle(u,eng_w,v0,ws,delta_t,gb_op): #ws  and gb_op are lists of len 2
  if u > 0:
    eng_tq= car.eng_wot(eng_w) * u/100
  else:   
    eng_tq= car.eng_wot(eng_w)
  gb_rat, gb_eff = car.g_box(v0)
  gear= car.gb[int(gb_rat)]
  gb_opt=eng_tq * gear * gb_eff- (car.Igb_o + car.Igb_i) * (gb_op[1] - gb_op[0])/delta_t
  wh_spt= gb_opt * car.Fdr * car.Fef- (car.wh_inf + car.wh_inr) * (ws[1] - ws[0])/delta_t
  wh_spd= odeint(car.roadload, ws[1], [0,delta_t], args=(wh_spt,u))
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

# set up the objective
last = np.zeros(num_points)
last[-1] = 1
final = mpc.Param(value=last)


# set up the variables
x = mpc.SV(value=0, lb=0, ub=x_goal,name='x')
v = mpc.SV(value=0, lb=0, ub=speed_limit, name='v') #vs
a = mpc.SV(value=0, ub=5, lb=-5,name='a')          #acc

grade=mpc.Param(value=0,name='grade')
#grade.STATUS=0
#grade.FSTATUS=1


eng_tq= mpc.SV(value=0,name='eng_tq')
gb_rat= mpc.SV(value=0,name='gb_rat')
gb_eff= mpc.SV(value=car.ge[0],name='gb_eff')
gear= mpc.SV(value=0,name='gear')
gb_opt= mpc.SV(value=0,name='gb_top')
wh_spt= mpc.SV(value=0,name='wh_spt')
ws= mpc.SV(value=0,name='ws')
gb_op= mpc.SV(value=0,name='gb_op')
gb_ip= mpc.SV(value=0,name='gb_ip')
eng_w= mpc.SV(value=car.eng_wmin,lb=car.eng_wmin,ub=car.eng_wmax,name='eng_w')
eng_wot=mpc.Var(name='eng_wot') #create interpolator variable for eng_wot

# set up the Manipulated Variables
ac_ped = mpc.MV(value=0, lb=0, ub=100,name='ac_ped') #+u
ac_int=mpc.SV(value=0,name='ac_int')

br_ped = mpc.MV(value=0, lb=0, ub=100,name='br_ped') #-u

brmac=mpc.Intermediate(br_ped-ac_ped,name='brmac') #brake-acc
u=mpc.sign2(brmac) #returns 1 if brake is on or -1 if accelerate
bsend=mpc.Intermediate((u+1)/2,name='bsend') #1 if brake, 0 if accelerate
asend=mpc.Intermediate((u-1)/-2,name='asend') #0 if brake, 1 if accelerate

vadj=mpc.Intermediate(v-0.1,name='vadj')
vsign=mpc.sign2(vadj) #1 if v>0.1 -1 if v<0.1
vlowsend=mpc.Intermediate((vsign-1)/-2,name='vlowsend') #1 if v<-.1, 0 if v>-0.1
vhighsend=mpc.Intermediate((vsign+1)/2,name='vhighsend') #0 if v<-.1, 1 if v>-0.1

# set up the gears (1 is in the gear, 0 is not in the gear)
vchks=[mpc.Intermediate(v-gcv[i]) for i in range(4)]
vsigns=[mpc.sign2(vchk)for vchk in vchks]

in_gr = [mpc.MV(integer=True, value=0, lb=0, ub=1,name='in_gr{0}'.format(i)) for i in range(6)] #gb_rat
in_gr[0].VALUE=1

gear_ratio = mpc.SV(value=car.ge[0],name='gear_ratio') #gb_eff

# set up the time variable (to minimize)
tf = mpc.FV(value=100, lb=1, ub=max_time,name='tf') #s

# turn them 'on'
for s in (ac_ped, br_ped, tf):
	s.STATUS = 1
# turn on the gears
for s in in_gr:
	s.STATUS = 1

# I'm going to assume that car's fuel pump is a plunger type,
# so the fuel rate is proportional to the %pedal
# This means I can minimize fuel usage by minimizing the ac_ped variable

# add stops
mpc.fix(x, num_points-1, x_goal) # destination

#if (stop_sign == "yes"):
#	# stop sign
#	mpc.fix(v, int(num_points/3), 0) 
#interpolators

mpc.cspline(eng_w,eng_wot,car.eng_spd,car.eng_trq,True)

# set up the governing equations for the car
mpc.Equation(eng_tq==eng_wot * ac_ped/100+eng_wot*bsend)    
    
# don't use break and accelerator at the same time
mpc.Equation(ac_ped * br_ped == 0)

# set up the gear logic (pick 1 of 4)
mpc.Equation(v**in_gr[0]*in_gr[0]**v == 0)# in gear 0 when v =0
mpc.Equation(in_gr[1] * (v-gcv[0])*(-in_gr[0]*2+1) <= 0)# in gear 1 when v < gcv[0]
mpc.Equation(in_gr[2] * (v-gcv[0]) * (v-gcv[1]) <= 0) # in gear 2 when v < gcv[1]
mpc.Equation(in_gr[3] * (v-gcv[1]) * (v-gcv[2]) <= 0) # in gear 3 when v > gcv[2]
mpc.Equation(in_gr[4] * (v-gcv[2]) * (v-gcv[3]) <= 0) # in gear 4 when v > gcv[2]
mpc.Equation(in_gr[5] * (v-gcv[3]) >= 0) # in gear 5 when v > gcv[3]

mpc.Equation(in_gr[0] + in_gr[1] + in_gr[2]+ in_gr[3]+ in_gr[4]+ in_gr[5] == 1) # must be in a gear
mpc.Equation(gb_rat==in_gr[0]*0 + in_gr[1]*1 + in_gr[2]*2+ in_gr[3]*3+ in_gr[4]*4+ in_gr[5]*5)

# set the gear ratio based on the gear
mpc.Equation(gb_eff == car.ge[0]*in_gr[0] + car.ge[1]*in_gr[1] + car.ge[2]*in_gr[2]+ car.ge[3]*in_gr[3]\
             + car.ge[4]*in_gr[4]+ car.ge[5]*in_gr[5])
mpc.Equation(gear==car.gb[0]*in_gr[0] + car.gb[1]*in_gr[1] + car.gb[2]*in_gr[2]+ car.gb[3]*in_gr[3]\
             + car.gb[4]*in_gr[4]+ car.gb[5]*in_gr[5])
mpc.Equation(gb_opt==eng_tq * gear * gb_eff)#- (car.Igb_o + car.Igb_i) * (gb_op.dt()/tf))#[1] - gb_op[0])/delta_t)

mpc.Equation(wh_spt== gb_opt * car.Fdr * car.Fef)#- (car.wh_inf + car.wh_inr) * (ws.dt()/tf))#[1] - ws[0])/delta_t)
#mpc.Equation(ws.dt()/tf==(1/car.Iw * (wh_spt - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - \
#             car.wh_rd*car.Crr*(car.m+car.load)*mpc.cos(grade)*ws - car.wh_rd*(car.m+car.load)*9.81*mpc.sin(grade)))*asend+\
#(1/car.Iw * (car.Fb*br_ped*car.wh_rd - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - car.wh_rd*car.Crr*(car.m+car.load)\
#            *mpc.cos(grade)*ws - car.wh_rd*(car.m+car.load)*9.81*mpc.sin(grade)))*bsend*vhighsend+\
#             (1/car.Iw * ( - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - car.wh_rd*car.Crr*(car.m+car.load)\
#            *mpc.cos(grade)*ws - car.wh_rd*(car.m+car.load)*9.81*mpc.sin(grade)))*bsend*vlowsend)
mpc.Equation(ws.dt()/tf==(1/car.Iw * (wh_spt - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - \
             car.wh_rd*car.Crr*(car.m+car.load)*ws))*asend+\
(1/car.Iw * (car.Fb*br_ped*car.wh_rd - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - car.wh_rd*car.Crr*(car.m+car.load)\
            *ws ))*bsend*vhighsend+\
             (1/car.Iw * ( - 0.5*car.rho*car.Cd*car.A*car.wh_rd**3*ws**2 - car.wh_rd*car.Crr*(car.m+car.load)\
            *ws))*bsend*vlowsend)
mpc.Equation(gb_op==ws*car.Fdr)
mpc.Equation(gb_ip==gb_op* gear)

mpc.Equation(eng_w==gb_ip* 60 / (2 * np.pi)) #eng_sp
mpc.Equation(v== ws * car.wh_rd)

mpc.Equation(x.dt() / tf == v)
mpc.Equation(v.dt() / tf == a)
mpc.Equation(ac_int.dt()/tf==ac_ped)


# set up the solver
mpc.options.IMODE = 6
mpc.options.SOLVER = 1


# set up the objective
mpc.Obj(100*ac_int*final + tf)# + 1000*(1-in_gr_1-in_gr_2-in_gr_3))
#%% Simulate
test_data=[[0]]*6 #[x,v,a,gear,acc,bra]

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
for ind,gr in enumerate(in_gr):
  plt.plot(time, np.array(gr.value),'o',label='Gear {0}'.format(ind))
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


