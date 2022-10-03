import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit
import glob
nside = 4
from mpl_toolkits import mplot3d
from matplotlib import ticker
import healpy as hp
import matplotlib
import copy
from matplotlib import rc
import sys
sys.path.insert(0, '../src/')
from AMC_Density_Evolution import *
from scipy import ndimage

import warnings
warnings.filterwarnings("ignore")


tList = []
topDir = "/Users/samuelwitte/Dropbox/Magnetized_Plasma/Axion_Infall/notebooks/data_paper/"
fileB = "Minicluster__MassAx_2.6e-5_AxG_1.0e-14_ThetaM_0.2_rotPulsar_1.67_B0_1.6e14_rNS_10.0_MassNS_1.0_Ntrajs_5000000_NS_Mag_0.00033_NS_Theta_0.0_Mmc_1.0e-12_Rmc_1.86e9__trace_trags__thetaCN__fixed_time"
fileT = "__NFW__.npz"
fileList = glob.glob(topDir + fileB + "*" + fileT)
for i in range(len(fileList)):
    tag1 = fileList[i].find("fixed_time_")
    tag2 = fileList[i].find("__NFW")
    timeT = float(fileList[i][tag1 + len("fixed_time_"):tag2])
    tList.append(timeT)
print(fileList)
print(tList)
thetaL = [0.3, 0.5, 0.8]
eps_th = 0.1
eps_phi = 0.07
b_param = np.asarray([0.0, 1.0e8, 0.0])
omega_rot = 1.67
mass=1.0e-5
NS_vel_T = 0.0
is_axionstar=False
plot_smoothed=True
tag = ""
phi_PT = 0.0
yERR = 2.0 * np.array([0.04, 0.069, 0.157])
time_MIN = -np.pi
time_MAX = np.pi
# time_evol_map_comp(fileList, thetaL, tList, eps_th, eps_phi, b_param, omega_rot=omega_rot, mass=mass, NS_vel_T=NS_vel_T, is_axionstar=False, tag="", sve=False, remove_dephase=True, yERR=0.2)

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['lines.dashed_pattern'] = 2.8, 1.5
mpl.rcParams['lines.dashdot_pattern'] = 4.8, 1.5, 0.8, 1.5
mpl.rcParams['lines.dotted_pattern'] = 1.1, 1.1
mpl.rcParams['lines.scale_dashes'] = True

# Default colors
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler('color',['cornflowerblue','forestgreen','maroon','goldenrod','firebrick','mediumorchid', 'navy', 'brown'])


# Fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.sans-serif'] = 'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['text.usetex'] = True

# Axes
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.labelpad'] = 8.0
mpl.rcParams['figure.constrained_layout.h_pad'] = .2

# Tick marks - the essence of life
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['xtick.minor.width'] = 0.75
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.labelsize'] = 20#22

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 0.75
mpl.rcParams['ytick.major.pad'] = 8
mpl.rcParams['ytick.labelsize'] = 20#22
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.minor.visible'] = True

# Legend
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
#mpl.rcParams['legend.edgecolor'] = 'black'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderpad'] = 0.4 # border whitespace
mpl.rcParams['legend.labelspacing'] = 0.5 # the vertical space between the legend entries
mpl.rcParams['legend.handlelength'] = 1.5 # the length of the legend lines
mpl.rcParams['legend.handleheight'] = 0.7 # the height of the legend handle
mpl.rcParams['legend.handletextpad'] = 0.5 # the space between the legend line and legend text
mpl.rcParams['legend.borderaxespad'] = 0.5 # the border between the axes and legend edge
mpl.rcParams['legend.columnspacing'] = 2.0 # column separation


# Figure size
mpl.rcParams['figure.figsize'] = 10, 6

# Save details
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

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


@jit(nopython=True)
def ThetaPhi(data):
    Tkf = data[:, 0]
    Pkf = data[:, 1]
    TXf = data[:, 2]
    PXf = data[:, 3]
    rX = data[:, 4]

    Theta = np.zeros(len(rX))
    Phi = np.zeros(len(rX))
    #r_proj = np.max(rX)
    r_proj = 1e10
    for i in range(len(rX)):
        xx = np.array([rX[i] * np.cos(PXf[i]) * np.sin(TXf[i]), rX[i] * np.sin(PXf[i]) * np.sin(TXf[i]), rX[i] * np.cos(TXf[i])])
        vv = np.array([ np.cos(Pkf[i]) * np.sin(Tkf[i]), np.sin(Pkf[i]) * np.sin(Tkf[i]), np.cos(Tkf[i])])
        tproj = -np.sum(xx*vv) + np.sqrt(4*np.sum(xx*vv)**2. - 4 * (rX[i]**2. - r_proj**2.))/2.
        Phi[i] = np.arctan2((vv[1] * tproj + xx[1]), (vv[0] * tproj + xx[0]))
        Theta[i] = np.arccos((vv[2] * tproj + xx[2]) / r_proj)
    return Theta,Phi

def diff_power_curve(arrayVals, ThetaVals, mass, period, binTot=300, eps_theta=0.01, dOM=False):
    thetaL = np.linspace(1e-2, np.pi-1e-2, binTot)
    rate = np.zeros(binTot)
    for i in range(len(thetaL)):
        new_arr = theta_cut(arrayVals, ThetaVals, thetaL[i], eps=eps_theta)
        if (thetaL[i] - eps_theta) < 0.0:
            thetaN = (thetaL[i] + eps_theta) / 2
            epsN = (thetaL[i] + eps_theta) - thetaN
        elif (thetaL[i] + eps_theta) > np.pi:
            thetaN = np.pi - (np.pi - (thetaL[i] - eps_theta)) / 2
            epsN = np.pi - thetaN
        else:
            thetaN = thetaL[i]
            epsN = eps_theta
            
        rate[i] = np.sum(new_arr[:,5]) * mass / ( np.sin(thetaN) * 2 * epsN) / period  # eV / s

    return np.column_stack((thetaL, rate))


def time_evol_map_comp(fileList, thetaL, tList, eps_th, eps_phi, b_param, omega_rot = 1.0, mass=1e-5, NS_vel_T=0.0, is_axionstar=False, tag="", sve=False, remove_dephase=True, yERR=[0.2], phi_PT=0.0, time_MIN=-np.pi, time_MAX=np.pi, plot_smoothed=True):
    # timeEvo = np.empty(len(thetaL), len(thetaL), dtype=object)

    fig, ax = plt.subplots(figsize=(10,6))
    colorL = ["#6B0504", "#94849B","#73AB84", "#79C7C5",  "#FFCAAF"]
    for i in range(len(fileList)):
        fileN = fileList[i]
        time = tList[i]
        
        phiV = omega_rot * time
        xpt0 = phiV - 2 * np.pi
        xptL = [xpt0]
        foundL = False
        foundH = False
        cnt = 1
        while not foundL:
            xpt_hold = xpt0 - 2*np.pi*cnt
            if xpt_hold >= time_MIN:
                xptL.append(xpt_hold)
                cnt +=1
            else:
                foundL = True
        cnt = 1
        while not foundH:
            xpt_hold = xpt0 + 2*np.pi*cnt
            if xpt_hold <= time_MAX:
                xptL.append(xpt_hold)
                cnt +=1
            else:
                foundH = True
                
        for jk,xpt in enumerate(xptL):
            file_use, den = eval_density_3d(fileN, b_param, xpt, NS_vel_T, is_axionstar=is_axionstar)
            file_use[:,5] *= den

            if remove_dephase:
                file_use[:, 5] /= file_use[:,15] **2
                file_use[:,15] = 1.0
            file_use[file_use[:,7] * file_use[:,15] > 1, 5] = 0.0

            ThetaVals = file_use[:,2]
            
            phi_big = np.linspace(-np.pi,np.pi,200)
            holdV = np.zeros(len(phi_big))

            for j,thetaC in enumerate(thetaL):
                file_short = theta_cut(file_use, ThetaVals, thetaC, eps=eps_th)
                Phi_short = file_short[:,3]
                Theta_short = file_short[:,2]
                # print(np.max(Phi_short), np.min(Phi_short))
               
                filePhi = phi_cut(file_short, Phi_short, phi_PT, eps=eps_phi)
                val = np.sum(filePhi[:,5]) * mass / ( np.sin(thetaC) * 2 * eps_th * 2 * eps_phi)  # eV / s

                if not is_axionstar:
                    if xpt < -np.pi:
                        xpt += 2*np.pi
                if jk == 0 and i == 0:
                    plt.errorbar(xpt, val, xerr=eps_phi, yerr=val*yERR[j], elinewidth=1, capsize=0.2, capthick=1, fmt='o', color=colorL[j], label=r"$\theta =${:.2f}".format(thetaC))
                else:
                    plt.errorbar(xpt, val, xerr=eps_phi, yerr=val*yERR[j], elinewidth=1, capsize=0.2, capthick=1, fmt='o', color=colorL[j])
                
                if np.abs(np.pi - xpt) / np.pi < 0.01:
                    plt.errorbar(-np.pi, val, xerr=eps_phi, yerr=val*yERR[j], fmt='o', color=colorL[j])

                if time == 0 and jk==0 and plot_smoothed:
                    for k in range(len(phi_big)):
                        filePhi = phi_cut(file_short, Phi_short, phi_big[k], eps=eps_phi)
                        
                        holdV[k] = np.sum(filePhi[:,5]) * mass / ( np.sin(thetaC) * 2 * eps_th * 2 * eps_phi)  # eV / s
                    
                    rateVs = ndimage.uniform_filter(holdV, size=15)
                    
                    plt.plot(phi_big, rateVs, c=colorL[j], label=r"$\theta =${:.2f}".format(thetaC))
    plt.yscale("log")
    plt.xlim([time_MIN, time_MAX])
    plt.xlabel(r'$\omega \times t$', fontsize=20);
    plt.ylabel('Flux [arb. units]', fontsize=20);
    # plt.xticks([0, 1, 2, 3, 4, 5, 6], ('0', '', '2', '', '4', '', '6'))
    ax.tick_params(direction='in', length=8, width=1, labelsize=18)#, colors='r',grid_color='r', grid_alpha=0.5)
    ax.tick_params(which='minor', direction='in', length=3, width=1, labelsize=12)
    ax.legend(fontsize=16)
    plt.savefig("plots_paper/TimeProjections_"+tag+".png", dpi=200)
    return

time_evol_map_comp(fileList, thetaL, tList, eps_th, eps_phi, b_param, omega_rot=omega_rot, mass=mass, NS_vel_T=NS_vel_T, is_axionstar=is_axionstar, tag=tag, sve=False, remove_dephase=True, yERR=yERR, phi_PT=phi_PT, time_MIN=time_MIN, time_MAX=time_MAX, plot_smoothed=plot_smoothed)
