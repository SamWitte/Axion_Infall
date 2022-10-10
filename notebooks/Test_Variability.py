import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit
import glob
from scipy import ndimage
import sys
sys.path.insert(0, '../src/')
from AMC_Density_Evolution import *

tList = []
topDir = "/Users/samuelwitte/Dropbox/Magnetized_Plasma/Axion_Infall/notebooks/data_paper/"
fileB = "Minicluster__MassAx_2.6e-5_AxG_1.0e-14_ThetaM_0.2_rotPulsar_1.67_B0_1.6e14_rNS_10.0_MassNS_1.0_Ntrajs_5000000_NS_Mag_0.00033_NS_Theta_0.0_Mmc_1.0e-12_Rmc_1.86e9__trace_trags__thetaCN__fixed_time_0.0"
fileT = ".npz"
fileList = glob.glob(topDir + fileB + "*" + fileT)
b_param = np.array([0.0, 0.0, 0.0])
NS_vel_T = 0.0
is_axionstar=False
thetaL = [0.3, 0.5, 0.9]
eps_th = 0.1
eps_phi = 0.07
mass = 1.0e-5
remove_dephase = True

def theta_cut(arrayVals, Theta, thetaV, eps=0.01):
    if (thetaV - eps) < 0:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > np.pi + (thetaV - eps)
        jointC = condition1
    elif (thetaV - eps) > np.pi:
        condition1 = Theta < (thetaV + eps) - np.pi
        condition2 = Theta > (thetaV - eps)
        jointC = Theta > (thetaV - eps)
    else:
        condition1 = Theta < (thetaV + eps)
        condition2 = Theta > (thetaV - eps)
    
        jointC = np.all(np.column_stack((condition1, condition2)), axis=1)
    return arrayVals[jointC]

def phi_cut(arrayVals, Phi, phiV, eps=0.01):
    if (phiV - eps) < -np.pi:
        condition1 = Phi < phiV + eps
        condition2 = Phi > 2 * np.pi + phiV - eps
        jointC = np.any(np.column_stack((condition1, condition2)), axis=1)
    elif (phiV + eps) > np.pi:
        condition1 = Phi < phiV + eps - 2 * np.pi
        condition2 = Phi > phiV - eps
        jointC = np.any(np.column_stack((condition1, condition2)), axis=1)
    else:
        condition1 = Phi < phiV + eps
        condition2 = Phi > phiV - eps
        jointC = np.all(np.column_stack((condition1, condition2)), axis=1)
    return arrayVals[jointC]

phi_big = np.linspace(-np.pi,np.pi,10)
final_hold = np.zeros((len(fileList), len(thetaL), len(phi_big)))

for i in range(len(fileList)):
    fileN = fileList[i]
    
    file_use, den = eval_density_3d(fileN, b_param, 0.0, NS_vel_T, is_axionstar=is_axionstar)
    file_use[:,5] *= den
    if remove_dephase:
        file_use[:, 5] /= file_use[:,15] **2
        file_use[:,15] = 1.0
    file_use[file_use[:,7] * file_use[:,15] > 1, 5] = 0.0

    ThetaVals = file_use[:,2]
    
    
    holdV = np.zeros(len(phi_big))

    #normF = np.sum(file_use[:, 5])
    normF = 1.0
    for j,thetaC in enumerate(thetaL):
        file_short = theta_cut(file_use, ThetaVals, thetaC, eps=eps_th)
        Phi_short = file_short[:,3]
        Theta_short = file_short[:,2]
        # print(np.max(Phi_short), np.min(Phi_short))
        
     
        for k in range(len(phi_big)):
            filePhi = phi_cut(file_short, Phi_short, phi_big[k], eps=eps_phi)
            holdV[k] = np.sum(filePhi[:,5]) * mass / ( np.sin(thetaC) * 2 * eps_th * 2 * eps_phi) / normF # eV / s
        
        final_hold[i, j, :] = ndimage.uniform_filter(holdV, size=1)
        

for j in range(len(thetaL)):
    print(thetaL[j])
    vals = []
    for k in range(len(phi_big)):
        # valC = np.abs(final_hold[:, j, k] - np.mean(final_hold[:, j, k])) / np.mean(final_hold[:, j, k])
        valC = (final_hold[:, j, k]) / np.mean(final_hold[:, j, k])
        # vals.append(np.std(final_hold[:, j, k]) / np.mean(final_hold[:, j, k]))
        # vals.append(np.std(final_hold[:, j, k]))
        vals.append(np.sqrt(2) * np.std(valC))
        
    valsT = np.asarray(vals)
    print(np.max(vals))
    # print(vals)
