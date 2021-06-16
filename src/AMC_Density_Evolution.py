import numpy as np

def AMC_CrossingTime(b, M, v_NS, rho_amc):
    # assume b [km], M [solar mass], v_NS [vector unitless]
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    delD = np.sqrt(R_amc**2 - b**2)  # b in km
    return 2 * delD / np.sqrt(np.sum(v_NS**2)) / 2.998e5 # s

def AMC_profile(r, M, rho_amc):
    # rho_amc = M/pc^3
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    c = 100
    fnwf = np.log(1+c) - c / (1+c)
    rs = (M / (4*np.pi * rho_amc * fnwf))**(1/3) * 3.086*10**13 # km
    
    rho_0 = rho_amc * (R_amc / rs)**(9.0/4.0) / 4.0
    
    if r < R_amc:
        density = rho_0 * (rs/r) ** (9.0 / 4.0)
    else:
        density = 0.0
    density *= 37.96
    return density # units: GeV / cm^3

def AMC_DensityEval(b, M, v_NS, t, rho_amc):
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    delD = np.sqrt(R_amc**2 - b**2)  # b in km
    vel = np.sqrt(np.sum(v_NS**2)) * 2.998e5
    r = np.sqrt((delD - vel * t)**2 + b**2)
    density = AMC_profile(r, M, rho_amc)
    return density
    
    
def Transient_Time(b, r_amc, rho_amc, v_NS):
    # assume b [km], v_NS [vector unitless]
    
    delD = np.sqrt(r_amc**2 - b**2)  # b in km
    return 2 * delD / v_NS / 2.998e5 # s

def Transient_AMC_DensityEval(b, r_amc, rho_amc, v_NS, t, nfw=True):
    # t in s?
    delD = np.sqrt(r_amc**2 - b**2)  # b in km
    vel = np.sqrt(np.sum(v_NS**2)) * 2.998e5 # km /s
    r = np.sqrt((delD - vel * t)**2 + b**2)
    density = transient_profile(r, r_amc, rho_amc, b, t, nfw=nfw)
    return density

    
def transient_profile(r, r_amc, rho_amc, b, t, nfw=True):
    if r > r_amc:
        return 0.0
    if nfw:
        c=100
        rs = r_amc / c
        den = rho_amc / (r/rs) / (1+r/rs)**2
    else:
        den = rho_amc / 4 * (r_amc / r)**(9/4)
    return den
