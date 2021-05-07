import numpy as np

def AMC_CrossingTime(b, M, v_NS):
    # assume b [km], M [solar mass], v_NS [vector unitless]
    R_amc = 1.4e8 * (M/1e-10) **(1./3.) # R in km
    delD = np.sqrt(R_amc**2 - b**2)  # b in km
    return 2 * delD / np.sqrt(np.sum(v_NS**2)) / 2.998e5 # s

def AMC_DensityEval(b, M, v_NS, t):
    delD = np.sqrt(R_amc**2 - b**2)  # b in km
    vel = np.sqrt(np.sum(v_NS**2)) * 2.998e5
    return np.sqrt((delD - vel * t)**2 + b**2)
    
