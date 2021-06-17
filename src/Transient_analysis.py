import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from AMC_Density_Evolution import *
import glob
import healpy as hp
import os

NFW = True
nside = 8

# run through each mass and each NS, determine coupling for which this would be observable

def sense_compute(mass, bwdith=1e-4, t_obs=1, SNR=5):
    # t_obs days, bwidth fractional
    SEFD = 0.098*1e3 #mJy
    return SNR * SEFD / np.sqrt(2 * mass * bwidth * t_obs * 24 * 60**2 / 6.58e-16)
    
def Find_Ftransient(NFW=True, nside=8, t_obs=1):
    # t_obs in days

    if NFW:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_NFW_AScut_wStripping.txt')
    else:
        orig_F = np.loadtxt('../encounter_data/Interaction_params_PL_AScut_wStripping.txt')
        
    AxionMass = [1.0e-6, 5.0e-6, 1.0e-5, 3.0e-5] # eV
    glist = np.zeros(len(AxionMass), dtype=object)
    for i in range(len(AxionMass)):
        glist[i] = []
        
    files = glob.glob('results/Minicluster_PeriodAvg*')
    # cycle through output files
    for i in range(len(files)):
        find1 = files[i].find('MassAx_')
        find2 = files[i].find('_ThetaM_')
        
        axM = float(files[i][find1+len('MassAx_'):find2])
        indx = -1
        # identify mass so we know how to store
        for j in range(len(AxionMass)):
            if axM == AxionMass[j]:
                indx = j
        if indx == -1:
            print('Problem with mass!', axM)
            break
        
        # identify NS in original file
        find1 = files[i].find('_rotPulsar_')
        find2 = files[i].find('_B0_')
        periodN = (2 * np.pi) / float(files[i][find1+len('_rotPulsar_'):find2])
        
        find1 = files[i].find('_B0_')
        find2 = files[i].find('_rNS_')
        B0 = float(files[i][find1+len('_B0_'):find2])
        
        possible = np.where(np.round(orig_F[:, 6], 3) == round(periodN, 3))[0]
        NSIndx = np.where(np.round(orig_F[:,7][possible], 2) == round(B0, 2))[0]
        print(possible)
        print(NSIndx)
        print(B0, periodN)
        print(orig_F[:,7][possible])
        dist = orig_F[NSIndx, 0] # pc
        dens_amc = orig_F[NSIndx, 3] # M/pc^3
        rad_amc = orig_F[NSIndx, 4] # pc
        bparam = orig_F[NSIndx, 5] # pc
        vel = orig_F[NSIndx, -1] * 3.086*10**13 / 2.998e5 # unitless
        dens_amc *= 3.8 * 10**10 # eV/cm^3
        rad_amc *= 3.086*10**13 # km
        bparam *= 3.086*10**13 # km
        
        
        # compute flux density
        file_in = np.load(files[i])
        Theta = file_in[:,2]
        Phi = file_in[:,3]
        
        pixel_indices = hp.ang2pix(nside, Theta, Phi)
        indxs = hp.nside2npix(nside)
        viewA = int(np.random.rand(1) * indxs)
        rel_rows = file_in[pixel_indices == viewA]
        
        rate = np.sum(rel_rows[:, 5])  / hp.pixelfunc.nside2resol(nside) # missing rho [eV / cm^3], will be in [eV / s]
        
        t_shift = t_obs / 2 * 24 * 60**2 # seconds
        t_mid = Transient_Time(bparam, rad_amc, vel) / 2
        tlist = np.linspace(t_mid - t_shift, t_mid + t_shift, 200)
        dense_scan = np.zeros_like(tlist)
        for j in range(len(tlist)):
            dense_scan[j] = Transient_AMC_DensityEval(bparam, rad_amc, dens_amc, vel, tlist[j], nfw=NFW)
            
        # print(dense_scan, dist)
        rate *= np.trapz(dense_scan, tlist) * (1 / (dist * 3.086*10**18))**2 * 1.6022e-12 # erg / s / cm^2
        bwidth = axM * 1e-4 / 6.58e-16 # Hz
        rate *= (1/bwidth) * 1e26 # mJy
        glim = np.sqrt(sense_compute(axM, bwdith=bwdith, t_obs=t_obs, SNR=5)  / rate) * 1e-12 # GeV^-1
        glist[indx].append(glim)

    if not os.path.isdir("amc_glims"):
        os.mkdir("amc_glims")
    for i in range(len(AxionMass)):
        fileN = "amc_glims/glim_"
        if NFW:
            fileN += "NFW_"
        else:
            fileN += "PL_"
        fileN += "AxionMass_{:.2e}_".format(AxionMass[i])
        fileN += ".dat"
        np.savetxt(fileN, glist[i])
        
    return

Find_Ftransient(NFW=NFW, nside=nside)
