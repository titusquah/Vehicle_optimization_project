import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import odeint



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
Fdr   = 4.71               #final drive ratio
Fef   = 0.9604             #final drive ratio efficiency
wh_in = 0.4                #wheel inertia
gb    = np.array([1.0,3.65,2.15,1.45,1.0,0.83])
ge    = np.array([0.0,0.95,0.95,0.95,0.95,0.95])
ws    = 0.0

#engine data
eng_i   = 0.1
eng_spd = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000])
eng_trq = np.array([31.93184024, 43.84989124, 52.39157764, 58.77201955, 60.621201, 60.99103728, 59.97387807, 56.73770113, 50.7270955])
eng_brk = np.array([0, -1.619401501, -2.80112692, -3.588943867, -4.245457989, -4.639366462, -5.033274935, -5.252112976, -5.3834158])

#vehicle plant model
def roadload(ws,t,whl_t,u,grade):
    Iw    = ((m + load)*wh_rd**2) + wh_in
    if u >= 0:
        dw_dt = 1/Iw * (whl_t - 0.5*rho*Cd*A*wh_rd**3*ws**2 - wh_rd*Crr*(m+load)*np.cos(grade)*ws - wh_rd*(m+load)*9.81*np.sin(grade))
    else:
        dw_dt = 1/Iw * (Fb*u*wh_rd - 0.5*rho*Cd*A*wh_rd**3*ws**2 - wh_rd*Crr*(m+load)*np.cos(grade)*ws - wh_rd*(m+load)*9.81*np.sin(grade))
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

def eng_dyn(eng_sp, t,vart,Fc,gear):        
    dw_dt = (vart / eng_i) + (Fc * wh_rd * Fef * Fdr * gear)
    return dw_dt




