import numpy as np

def AMC_CrossingTime(b, M, v_NS, rho_amc):
    # assume b [km], M [solar mass], v_NS [vector unitless]
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    bnorm = np.sqrt(np.sum(b**2))
    delD = np.sqrt(R_amc**2 - bnorm**2)  # b in km
    return 2 * delD / np.sqrt(np.sum(v_NS**2)) / 2.998e5 # s

def AMC_profile(r, M, rho_amc):
    # rho_amc = M/pc^3
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    c = 100
    fnwf = np.log(1+c) - c / (1+c)
    rs = (M / (4*np.pi * rho_amc * fnwf))**(1/3) * 3.086*10**13 # km
    
    rho_0 = rho_amc * (R_amc / rs)**(9.0/4.0) / 4.0
    
    density = np.zeros(len(r))
    density[r < R_amc] = rho_0 * (rs / r[r < R_amc]) ** (9.0 / 4.0)
    # if r < R_amc:
    #     density = rho_0 * (rs/r) ** (9.0 / 4.0)
    # else:
    #    density = 0.0
    density *= 37.96
    return density # units: GeV / cm^3

def AMC_DensityEval(b, M, v_NS, t, rho_amc):
    R_amc = (3*M / (4*np.pi * rho_amc))**(1/3) * 3.086*10**13 #	km
    bnorm = np.sqrt(np.sum(b**2))
    delD = np.sqrt(R_amc**2 - bnorm**2)  # b in km
    vel = np.sqrt(np.sum(v_NS**2)) * 2.998e5
    r = np.sqrt((delD - vel * t)**2 + bnorm**2)
    density = AMC_profile(r, M, rho_amc)
    return density
    
    
def Transient_Time(b, r_amc, v_NS):
    # assume b [km], v_NS [vector unitless], rmac km
    bnorm = np.sqrt(np.sum(b**2))
    delD = np.sqrt(r_amc**2 - bnorm**2)  # b in km
    return 2 * delD / v_NS / 2.998e5 # s

def Transient_AMC_DensityEval(b, r_amc, rho_amc, v_NS, t, nfw=True):
    # t in s
    bnorm = np.sqrt(np.sum(b**2))
    delD = np.sqrt(r_amc**2 - bnorm**2)  # b in km
    vel = np.sqrt(np.sum(v_NS**2)) * 2.998e5 # km /s
    
    r = np.sqrt((delD - vel * t)**2 + bnorm**2)
    density = transient_profile(r, r_amc, rho_amc, b, t, nfw=nfw)
    return density

    
def transient_profile(r, r_amc, rho_amc, b, t, nfw=True):
    try:
        den = np.zeros_like(r)
        den[r>r_amc] = 0.0
        if nfw:
            c=100
            rs = r_amc / c
            den[r<=r_amc] = rho_amc / (r/rs) / (1+r/rs)**2
        else:
            den[r<=r_amc] = rho_amc / 4 * (r_amc / r)**(9/4)
    except:
        if r > r_amc:
            return 0.0
        if nfw:
            c=100
            rs = r_amc / c
            den = rho_amc / (r/rs) / (1+r/rs)**2
        else:
            den = rho_amc / 4 * (r_amc / r)**(9/4)
    
    return den

def get_NS_vel(fileN):
    fnm1 = fileN.find("_NS_Mag_")
    fnm2 = fileN.find("_NS_Theta_")
    fnm3 = fileN.find("_Mmc_")
    fnm4 = fileN.find("_Rmc_")
    fnm5 = fileN.find("___")
    
        
    NS_vel_M = float(fileN[fnm1 + len("_NS_Mag_"): fnm2])
    NS_vel_T = float(fileN[fnm2 + len("_NS_Theta_"): fnm3])
    MMc = float(fileN[fnm3 + len("_Mmc_"): fnm4])
    R_amc = float(fileN[fnm4 + len("_Rmc_"): fnm5])
    return NS_vel_T, NS_vel_M, MMc, R_amc

def get_t0_point(fileN, b, NS_vel_T_true):
    # b impact param [km]
    NS_vel_T, NS_vel_M, Mmc, R_amc = get_NS_vel(fileN)
    NS_vel = np.array([0.0, np.sin(NS_vel_T_true), np.cos(NS_vel_T_true)]) * NS_vel_M * 2.998e5 # km /s
    NS_hat = np.array([0.0, np.sin(NS_vel_T_true), np.cos(NS_vel_T_true)])
    
    Mass_NS = 1.0
    Roche_R = R_amc * (2 * Mass_NS / Mmc)**(1.0 / 3.0)
    # print(Roche_R)
    # z_comp = -np.sin(NS_vel_T) / np.cos(NS_vel_T)
    # perp_vec = np.array([1.0, 1.0, z_comp])
    # perp_vec /= np.sqrt(np.sum(perp_vec**2))
    
    # initial position of center of AMC at initial infall
    init_pos = b + (-NS_hat) * Roche_R
    
    fileData = np.load(fileN)
    rel_dist = np.sqrt(np.sum((init_pos - fileData[:,18:18+3])**2, axis=1))
    # dist_from_center = fileData[:, 18:18+3] + init_pos
    # dist_proj_vel = np.dot(dist_from_center, -NS_hat) + R_amc
    # closet_to_earth = np.argmin(np.sqrt(np.sum(fileData[:, 18:18+3]**2, axis=1)))
    # eff_time = ((np.max(dist_proj_vel) )/ (NS_vel_M * 2.998e5))  # s
    # dist_proj_vel =  + -NS_vel * eff_time # * eff_time_list[np.argmin(rad_dist)]
    # return init_pos, dist_proj_vel, NS_vel, Mmc, R_amc, fileData
    return init_pos, rel_dist, Mmc, R_amc, fileData

def eval_density_3d(fileN, b, t, NS_Vel_T):
    # given input file and impact parameter, return density-weighted raytracer output at time t
    # t [s]
    
    # init_pos, dist_proj, NS_vel, Mmc, R_amc, fileData = get_t0_point(fileN, b)
    # newDist_c = dist_proj - NS_vel * t
    # dist_from_center = fileData[:, 18:18+3] + init_pos
    init_pos, rel_dist, Mmc, R_amc, fileData = get_t0_point(fileN, b, NS_Vel_T)
    # print(init_pos, rel_dist)
    # rad_dist = np.sqrt(np.sum((dist_from_center - newDist_c)**2, axis=1)) # km
    # print(init_pos, dist_from_center, newDist_c, rad_dist)
    # print(np.min(rad_dist) / R_amc, np.max(rad_dist) / R_amc)
    rho_amc = (3*Mmc / (4*np.pi * (R_amc / 3.086e13)**3))  #
    
    den = AMC_profile(rel_dist, Mmc, rho_amc)
    # fileData[:, 5] *= den # properly re-weight density terms
    return fileData, den

