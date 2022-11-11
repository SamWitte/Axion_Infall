import numpy as np
import glob
from AMC_Density_Evolution import *
import random

freq = 8.0e9 #Hz
mass = 2*np.pi * freq * 6.58e-16 # eV
g_agg = 1e-14 # 1/GeV
tau = 100.0
deltV = 1.0
maxCM = 1.0
dm_OverD = 100.0 # extra dark matter density

def theta_cut(arrayVals, Theta, thetaV, eps=0.03):
    if (thetaV - eps) < 0:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > np.pi + (thetaV - eps)
        jointC = np.any(np.column_stack((condition1, condition2)), axis=1)
    elif (thetaV - eps) > np.pi:
        condition1 = Theta < (thetaV + eps) - np.pi
        condition2 = Theta > (thetaV - eps)
        jointC = np.any(np.column_stack((condition1, condition2)), axis=1)
    else:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > (thetaV - eps)
    
        jointC = np.all(np.column_stack((condition1, condition2)), axis=1)
    return arrayVals[jointC]
    
def diff_power_curve(arrayVals, ThetaVals, mass, period, binTot=300, eps_theta=0.01):
    thetaL = np.linspace(1e-4, np.pi-1e-4, binTot)
    rate = np.zeros(binTot)
    for i in range(len(thetaL)):
        new_arr = theta_cut(arrayVals, ThetaVals, thetaL[i], eps=eps_theta)
        rate[i] = np.sum(new_arr[:,5]) * mass / ( np.sin(thetaL[i]) * 2 * np.sin(eps_theta)) / period  # eV / s
    return np.column_stack((thetaL, rate))
    
def extract_file_params(fileN):
    tag1 = fileN.find('rotPulsar_') + len('rotPulsar_')
    tag2 = fileN.find('_B0_')
    period = 2*np.pi / float(fileN[tag1:tag2])
    
    tag1 = fileN.find('_MassAx_') + len('_MassAx_')
    tag2 = fileN.find('_AxG_')
    mass = float(fileN[tag1:tag2])
    
    tag1 = fileN.find('ThetaM_') + len('ThetaM_')
    tag2 = fileN.find('_rotPulsar')
    thetaM = float(fileN[tag1:tag2])
    
    tag1 = fileN.find('_B0_') + len('_B0_')
    tag2 = fileN.find('_rNS_')
    B0 = float(fileN[tag1:tag2])
    
    fft = "_Mmc_"
    tag1 = fileN.find(fft) + len(fft)
    tag2 = fileN.find('_Rmc_')
    M_amc = float(fileN[tag1:tag2])
    
    fft = "_Rmc_"
    tag1 = fileN.find(fft) + len(fft)
    tag2 = fileN.find('__trace')
    R_amc = float(fileN[tag1:tag2])
    
    fft = "_NS_Mag_"
    tag1 = fileN.find(fft) + len(fft)
    tag2 = fileN.find('_NS_Theta')
    v_amc = float(fileN[tag1:tag2])
    
    fft = "_NS_Theta_"
    tag1 = fileN.find(fft) + len(fft)
    tag2 = fileN.find('_Mmc_')
    v_Theta = float(fileN[tag1:tag2])
    
    return mass, period, thetaM, B0, M_amc, R_amc, v_amc, v_Theta
    
def test_plamsaF(B, P, mass, CM=1):
    op = 69.2 * np.sqrt(2) * np.sqrt(B / 1e14 / P) * 1e-6 * np.sqrt(CM) # eV
    if op  < mass:
        return False
    else:
        return True

def get_flux(mass, g_agg, binTot=200, eps_theta=0.03, bandwidth=90e3, dist=752, CM=1, tau=1.0, deltV=1, dm_OverD=1.0):
    # bandwidth in Hz
    # dist in kpc
    # eff_rate = 0.837425  # effective interaction rate / day -- NEED AUTO READ FILE! SJW
    

    dirN = "results"
    BradK_File = np.loadtxt("../../M31_encounter_data/M31_encounter_data_Oct2022/Interaction_params_PL_M_AMC_1.00e-14_M31_maxCM_{:.1f}_tauO_{:.1f}_delta_{:d}.txt".format(CM, tau, deltV))
    Rate_File = np.loadtxt("../../M31_encounter_data/M31_encounter_data_Oct2022/EncounterRates_M31.txt")
    cond1 = (Rate_File[:, 2] == CM)
    cond2 = (Rate_File[:, 3] == tau)
    
    eff_rate = Rate_File[np.all(np.column_stack((cond1, cond2)), axis=1), 4] # number / day
    eff_rate *= dm_OverD
    print("Effective Event Rate: {:.2e} \n".format(eff_rate[0]))
            
    
    properties_list = []
    cnt = 0
    for i in range(len(BradK_File)):
        B0, P, ThM, Age, xx, yy, zz, MC_Den, MC_Rad, MC_Mass, b, velNS = BradK_File[i, :]
        
        if test_plamsaF(B0, P, mass, CM=CM):
            properties_list.append([B0, P, ThM, Age, xx, yy, zz, MC_Den, MC_Rad, MC_Mass, b, velNS])
            cnt += 1
            
    # tot_NSs = cnt
    # tot_NSs = 10000
    properties_list = np.asarray(properties_list)
    
    fileList = glob.glob(dirN+"/*")
    flux_density_list = []
    # total_time_length = tot_NSs / eff_rate # days to simulate
    total_time_length = 90.0
    tot_NSs = int(eff_rate * total_time_length)
    
    observations = 5
    for j in range(observations):
        central_obs_window = random.random() * (total_time_length - 2.0 / 24.0) + 1.0 / 24.0 # days
    
        num_in_window = 0
        for idx in range(tot_NSs):
            
            i = idx % len(fileList)
            mass, period, thetaM, B0, MC_Mass, MC_R, velNS, v_Theta  = extract_file_params(fileList[i])
            
            indx_g = int(random.random() * len(properties_list[:,0]))
            
            b_param = properties_list[indx_g, 10]
            # MC_Mass = properties_list[i, 9]
            # MC_R = properties_list[i, 8]
            MC_Den = properties_list[indx_g, 7] * 37.96
            # velNS = properties_list[i, 11]
            
            transit_time = AMC_CrossingTime(b_param * 3.086e+13, MC_Mass, velNS, MC_R) # seconds
            time_sample = random.random() * total_time_length # peak encounter, days
            
            file_use, den = eval_density_3d(fileList[i], b_param * 3.086e+13, (time_sample - central_obs_window) / 1.15741e-5 , v_Theta, is_axionstar=False, is_nfw=False)
            if np.sum(den) > 0:
                num_in_window += 1
            else:
                continue
            
            
            file_use[:, 5] *= den
            prob = file_use[:,16]
            LZ_prob = (1.0 - np.exp(-prob * (g_agg / 1e-14)**2))
            file_use[:,5] *= LZ_prob / prob
            
            ThetaVals = file_use[:, 2]
            
            vals = diff_power_curve(file_use, ThetaVals, mass, period, binTot=binTot, eps_theta=eps_theta)
            randView = int(random.random() * binTot)
            flux_density = vals[randView, 1] / bandwidth / dist**2 * (1/3.086e+21)**2  * 1.60218e-12 # erg / ( s * Hz * cm^2)
            # flux_density = mass * np.sum(file_use[:,5]) / (4*np.pi) / bandwidth / dist**2 * (1/3.086e+21)**2  * 1.60218e-12 # erg / ( s * Hz * cm^2)
            flux_density *= 1.0e26 # mJy
            flux_density_list.append(flux_density)
    
    
        print("number in window for observing run {:.0f} is {:.0f} \n".format(j, num_in_window))
    print(flux_density_list)
    flux_density_list = np.asarray(flux_density_list)
    print("Maximum flux observed: {:.2e} mJy [computed with g_agg = {:.1e} 1/GeV]".format(np.max(flux_density_list), g_agg))
    print("Limit around 10 mJy... \t", np.sqrt(10 / np.max(flux_density_list)) * g_agg )

get_flux(mass, g_agg, CM=maxCM, deltV=deltV, tau=tau, dm_OverD=dm_OverD)
