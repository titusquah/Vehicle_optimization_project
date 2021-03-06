import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

from scipy import interpolate


fuel_df=pd.read_csv("Vehicle engine data - Sheet1.csv")
rpm=fuel_df['speed(rpm)'].values
nm=fuel_df['torque(Nm)'].values
fuel_flow=fuel_df['Fuel flow (l/hr.)'].values
rpm1,nm1=np.meshgrid(rpm, nm)
rpm1=rpm1.flatten()
nm1=nm1.flatten()
fuel_flow1=interpolate.griddata((rpm, nm), fuel_flow, (rpm1, nm1))
fuel_flow1=fuel_flow1.reshape(len(rpm),len(rpm))

rpm2,nm2=np.meshgrid(np.linspace(min(rpm),max(rpm),100),np.linspace(min(nm),max(nm),100))
tck=interpolate.bisplrep(rpm, nm, fuel_flow) # Build the spline


#Simulation time step definition
tf        = 300                 #final time for simulation
nsteps    = 3001                 #number of time steps
delta_t   = tf / (nsteps - 1)   #length of each time step
ts        = np.linspace(0,tf,nsteps)

#Vehicle data
m     = 300                #mass in Kg
load  = 60.0               #total passenger weight in kg
rho   = 1.19               #air density in kg/m^3
A     = 0.7                #area in m^2
Cd    = 0.5                #coefficient of drag dimensionless
Fp    = 30                 #engine power plant force
Fb    = 50                 #brake power plant force
Crr   = 0.005              #rolling resistance factor
wh_rd = 0.265              #dynamic rolling radius in m
Igb_i = 0.2                #gearbox input inertias
Igb_o = 0.2                #gearbox output inertias
Fdr   = 4.71               #final drive ratio
Fef   = 0.9604             #final drive ratio efficiency
wh_inr= 0.4                #wheel inertia rear
wh_inf= 0.35               #wheel inertia front
mdamp = 0.35
vdamp = 0.15
gb    = np.array([1.0,3.65,2.15,1.45,1.0,0.83])
ge    = np.array([0.0,0.95,0.95,0.95,0.95,0.95])
ws    = 0.0

#engine data
eng_i   = 0.1
eng_spd = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
eng_trq = np.array([31.93184024, 43.84989124, 52.39157764, 58.77201955, 60.621201, 60.99103728, 59.97387807, 56.73770113, 50.7270955])
eng_brk = np.array([0, -1.619401501, -2.80112692, -3.588943867, -4.245457989, -4.639366462, -5.033274935, -5.252112976, -5.3834158])

#variables assign
vs        = np.zeros(nsteps)   #variable for actual vehicle speed
acc       = np.zeros(nsteps)
wh_sp     = np.zeros(nsteps)
wh_spt    = np.zeros(nsteps)
gb_op     = np.zeros(nsteps)
gb_opt    = np.zeros(nsteps)
gb_ip     = np.zeros(nsteps)
gb_ipt    = np.zeros(nsteps)
gb_rat    = np.zeros(nsteps)
gb_eff    = np.zeros(nsteps)
eng_sp    = np.zeros(nsteps)
eng_tq    = np.zeros(nsteps)
eng_re    = np.zeros(nsteps)
es        = np.zeros(nsteps)
ies       = np.zeros(nsteps)
sp_store  = np.zeros(nsteps)
act_ped   = np.zeros(nsteps)
test      = np.zeros(nsteps)
v0        = 0.0                #variable for initial velocity
eng_wmin  = 1000
eng_wmax  = 9000
eng_w     = eng_wmin
eng_t     = 0
vgear     = 0.0
Iw    = ((m + load)*wh_rd**2) + wh_inf + wh_inr
#Drive cycle data
grade = 0                  #road grade factor

#vehicle plant model
def roadload(ws,t,whl_t,u,grade,v0):
    Iw    = ((m + load)*wh_rd**2) + wh_inf + wh_inr
    if u >= 0:
        dw_dt = 1/Iw * (whl_t - 0.5*rho*Cd*A*wh_rd**3*ws**2 - wh_rd*Crr*(m+load)*np.cos(grade)*ws - wh_rd*(m+load)*9.81*np.sin(grade))
    else:
        if v0 > 0.1:
            dw_dt = 1/Iw * (Fb*u*wh_rd - 0.5*rho*Cd*A*wh_rd**3*ws**2 - wh_rd*Crr*(m+load)*np.cos(grade)*ws - wh_rd*(m+load)*9.81*np.sin(grade))
        else:
            dw_dt = 1/Iw * (Fb*0*wh_rd - 0.5*rho*Cd*A*wh_rd**3*ws**2 - wh_rd*Crr*(m+load)*np.cos(grade)*ws - wh_rd*(m+load)*9.81*np.sin(grade))
    return dw_dt

#gear shift plant model & efficiency
def g_box(vgear,u):
    gvar = 0
    evar = 0
    if u > 0 or u < 0:
        gvar = 1
        evar = ge[gvar]
    else:
        gvar = 0
        evar = ge[gvar]
        
    if vgear > 0.2 and vgear <= 7.15111:
        gvar = 1
        evar = ge[gvar]

    elif vgear > 7.15111 and vgear <= 11.1736:
        gvar = 2
        evar = ge[gvar]

    elif vgear > 11.1736 and vgear <= 17.8778:
        gvar = 3
        evar = ge[gvar]

    elif vgear > 17.8778 and vgear <= 20.5594:
        gvar = 4
        evar = ge[gvar]

    elif vgear > 20.5594:
        gvar = 5
        evar = ge[gvar]

    return gvar, evar  

#engine wide open throttle torque table
def eng_wot(eng_w,u):

    if eng_w < 1000:
        eng_w = 1000
    if eng_w > 9000:
        eng_w = 9000
        
    for e in range (np.size(eng_spd)):
        esvar = eng_w - eng_spd[e]
        if esvar <= 1000:
            break
    if u > 0:
        etvar = eng_trq[e] + (eng_w - eng_spd[e]) * ((eng_trq[e] - eng_trq[e+1]) / (eng_spd[e] - eng_spd[e+1]))
    if u <= 0:
        etvar = eng_brk[e] + (eng_w - eng_spd[e]) * ((eng_brk[e] - eng_brk[e+1]) / (eng_spd[e] - eng_spd[e+1]))
    return etvar

def eng_dyn(eng_sp, t):        
    dw_dt = (vart / eng_i) + (Fc * wh_rd * Fef * Fdr * gear)
    return dw_dt

#Advanced cyber driver
step      = np.zeros(nsteps)   #assigning array for pedal position
#step[11:] = 75.0               #75% @ timestep 11
#step[40:] = -50                #-50% @ timestep 40 to simulate braking
ubias     = 0.0
kc        = 15.0
tauI      = 09.0
sum_int   = 0
sp        = 0
gear      = 1

v_new=np.linspace(0,50,500+1)
gb_new=[gb[int(g_box(v_new[i],10)[0])]for i in range(len(v_new))]
ge_new=[g_box(v_new[i],10)[1]for i in range(len(v_new))]
br_new=list(map(lambda hi: 0 if hi<0.1 else 1,v_new))
